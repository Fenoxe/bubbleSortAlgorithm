import cv2
import numpy as np
import skvideo.io
from tqdm import tqdm

class ImageLayer:

    CROP_FACTOR = 0.1

    COLOR_TO_LETTER = {
        (210, 118,  25): 'b',
        (  0, 140, 251): 'o',
        (158, 158, 158): 'g',
        (158, 103, 144): 'p',
        (247, 195,  79): 'l',
        (149, 101, 234): 'i',
        ( 51, 202, 192): 'd',
        ( 72,  85, 121): 'w',
        ( 62,  62, 255): 'r',
        ( 76, 209, 255): 'y',
        ( 83, 200,   0): 'e',
        (126,  35,  26): 'n',
        (107, 121,   0): 't',
    }

    LETTER_TO_COLOR = None

    BINS_TO_LAYOUT = {
         9: (5,4,0),
        11: (6,5,0),
        13: (7,6,0),
        15: (5,5,5),
    }

    OUTPUT_SIZE = (1200,600)
    PADDING = OUTPUT_SIZE[1] // 10
    BALL_RAD = 20

    def __init__(self, _input_dir, _output_dir):
        self.input_dir = _input_dir
        self.output_dir = _output_dir

        if ImageLayer.LETTER_TO_COLOR is None:
            ImageLayer.LETTER_TO_COLOR = {
                l:c for c,l in ImageLayer.COLOR_TO_LETTER.items()
            }   

    def parseImage(self, im_fn, num_bins):
        filepath = self.input_dir + im_fn

        im = cv2.imread(filepath)
        if im is None:
            print('could not find specified file name')
            return None
        
        im = im[int(im.shape[0] * ImageLayer.CROP_FACTOR) : int(im.shape[0] * (1 - ImageLayer.CROP_FACTOR))]

        circle_coords = self._getPossiblePoints(im)
        if circle_coords is None:
            print('couldnt find balls -> circle detection params need to be tuned')
            return None

        # self._displayPts(im, circle_coords)

        # choose correct points (params need tuning eventually)
        correct_points = self._getCorrectPoints(im, circle_coords, num_bins)
        if correct_points is None:
            print('could not choose correct points')
            return None

        # sort points into bins
        bin_pts = self._sortAndGroupPoints(im, correct_points, num_bins)
        if bin_pts is None:
            print('could not sort points into bins')
            return None

        # show user the result
        self._displayAugmentedBalls(im, bin_pts, num_bins)

        # build state from points
        state = self._getStateFromPoints(im, bin_pts, num_bins)
        if state is None:
            print('could not get state from points')
            return None

        return state

    def _getPossiblePoints(self, im):
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            im_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=300,
            param2=30,
            minRadius=23,
            maxRadius=27,
        )

        if circles is None:
            return None
            
        return np.around(circles).astype(np.uint16)[0,:,0:2]

    def _getCorrectPoints(self, im, circle_coords, num_bins):
        num_pts = 4 * (num_bins - 2)

        if circle_coords.shape[0] != num_pts:
            print(f'got {circle_coords.shape[0]} pts when there should be {num_pts} pts')
            return None
        
        return circle_coords

    def _sortAndGroupPoints(self, im, circle_coords, num_bins):
        num_full_bins = num_bins - 2
        num_pts = num_full_bins * 4

        if num_bins not in ImageLayer.BINS_TO_LAYOUT:
            print(f'bin layout not recognized')
            return None
        
        row1, row2, row3 = ImageLayer.BINS_TO_LAYOUT[num_bins]

        TWO_ROWS = (row2 != 0)
        THREE_ROWS = (row3 != 0)

        if not TWO_ROWS:
            print('single row solves not currently supported, solve it yo damn self lol')
            return None      

        # sort pts by x coord then by y coord, results in pts put into bins
        circle_coords = np.array([(x,y) for x,y in sorted(list(circle_coords), key=lambda t: (t[0],t[1]))])

        # reorganize pts to be indexed by bin
        bin_coords = circle_coords.reshape((num_full_bins,4,2))
        
        # set x coord of whole bin to be same
        bin_coords[:,:,0] = np.repeat(np.mean(bin_coords[:,:,0], axis=1).reshape(num_full_bins,1), 4, axis=1)

        # sort each bin individually by y coordinate
        bin_coords = np.array([sorted(b, key=lambda c: c[1], reverse=True) for b in bin_coords])

        # split top row and bottom row bins
        if THREE_ROWS:
            y_height = np.max(circle_coords[:,1]) - np.min(circle_coords[:,1])
            y_divider1 = np.min(circle_coords[:,1]) + (1 * y_height / 3)
            y_divider2 = np.min(circle_coords[:,1]) + (2 * y_height / 3)
            bin_coords = np.array([b for b in sorted(bin_coords, key=lambda b: int(np.mean(b[:,1]) > y_divider1) + int(np.mean(b[:,1]) > y_divider2))])
        elif TWO_ROWS:
            y_divider = 0.5 * (np.max(circle_coords[:,1]) + np.min(circle_coords[:,1]))
            bin_coords = np.array([b for b in sorted(bin_coords, key=lambda b: np.mean(b[:,1]) > y_divider)])

        """
            now, all coordinates are sorted in this way:

            bin_coords[0, 3]  bin_coords[1, 3]  ...  bin_coords[a, 3]
            bin_coords[0, 2]  bin_coords[1, 2]  ...  bin_coords[a, 2]
            bin_coords[0, 1]  bin_coords[1, 1]  ...  bin_coords[a, 1]
            bin_coords[0, 0]  bin_coords[1, 0]  ...  bin_coords[a, 0]

                    bin_coords[a+1, 3]  ...  bin_coords[b, 3]
                    bin_coords[a+1, 2]  ...  bin_coords[b, 2]
                    bin_coords[a+1, 1]  ...  bin_coords[b, 1]
                    bin_coords[a+1, 0]  ...  bin_coords[b, 0]

        """
        
        return bin_coords

    def _displayPts(self, im, pts):
        im_out = im.copy()

        for x,y in pts:
            coord = (x,y)
            cv2.circle(im_out, coord, 5, (0,0,0), thickness=-1)
        
        cv2.imshow('found pts', im_out)
        cv2.waitKey()

    def _displayAugmentedBalls(self, im, bin_pts, num_bins):
        num_full_bins = num_bins - 2
        num_pts = num_full_bins * 4

        im_out = im.copy()

        # get color for each point in each bin
        pts = bin_pts.reshape((num_pts,2))
        bin_colors = im[pts[:,1], pts[:,0]].reshape((num_full_bins, 4, 3))

        assert bin_colors.shape[0] == bin_pts.shape[0]
        assert bin_colors.shape[1] == bin_pts.shape[1]
        assert (bin_colors.shape[2] == 3) and (bin_pts.shape[2] == 2)

        for b_i in range(bin_pts.shape[0]):
            for p_i in range(bin_pts.shape[1]):
                coord = tuple(bin_pts[b_i,p_i].tolist())
                color = tuple(bin_colors[b_i,p_i].tolist())
                
                cv2.circle(im_out, coord, 20, (0,0,0), thickness=-1)
                cv2.circle(im_out, coord, 15, color, thickness=-1)
        
        cv2.imshow('augmented balls', im_out)
        cv2.waitKey()

    def _getStateFromPoints(self, im, bin_pts, num_bins):
        # makes the assumption that empty bins are all at the end

        num_full_bins = num_bins - 2
        num_pts = num_full_bins * 4

        # get color for each point in each bin
        pts = bin_pts.reshape((num_pts,2))
        bin_colors = im[pts[:,1], pts[:,0]].reshape((num_full_bins, 4, 3))

        assert bin_colors.shape[0] == bin_pts.shape[0]
        assert bin_colors.shape[1] == bin_pts.shape[1]
        assert (bin_colors.shape[2] == 3) and (bin_pts.shape[2] == 2)

        state = ''
        failed = False

        for b_i in range(bin_pts.shape[0]):
            for p_i in range(bin_pts.shape[1]):
                color = tuple(bin_colors[b_i,p_i].tolist())

                if color not in ImageLayer.COLOR_TO_LETTER:
                    print(f'could not find color {color} in color dict')
                    print(f'color={color}  bin={b_i+1} from bottom={p_i + 1} [both 1-indexed]')
                    failed = True
                    continue
                
                letter = ImageLayer.COLOR_TO_LETTER[color]

                state += letter

        if failed:
            return None

        state += ' ' * 8 # empty bins

        assert len(state) == 4 * num_bins

        return state

    def createSolPathVideo(self, filename, sol_path):
        num_frames = len(sol_path)
        num_bins = len(sol_path[0]) // 4
        vid = np.ones((num_frames, ImageLayer.OUTPUT_SIZE[0], ImageLayer.OUTPUT_SIZE[1], 3), dtype=np.uint8) * 255

        r1,r2,r3 = ImageLayer.BINS_TO_LAYOUT[num_bins]
        
        if r3:
            r1_x = np.linspace(int(ImageLayer.PADDING), int(ImageLayer.OUTPUT_SIZE[1] - ImageLayer.PADDING), num=r1, dtype=int)
            r2_x = np.linspace(int(ImageLayer.PADDING), int(ImageLayer.OUTPUT_SIZE[1] - ImageLayer.PADDING), num=r2, dtype=int)
            r3_x = np.linspace(int(ImageLayer.PADDING), int(ImageLayer.OUTPUT_SIZE[1] - ImageLayer.PADDING), num=r3, dtype=int)

            r1_y = np.linspace(int(2 * ImageLayer.OUTPUT_SIZE[0] // 7), int(1 * ImageLayer.OUTPUT_SIZE[0] // 7), num=4, dtype=int)
            r2_y = np.linspace(int(4 * ImageLayer.OUTPUT_SIZE[0] // 7), int(3 * ImageLayer.OUTPUT_SIZE[0] // 7), num=4, dtype=int)
            r3_y = np.linspace(int(6 * ImageLayer.OUTPUT_SIZE[0] // 7), int(5 * ImageLayer.OUTPUT_SIZE[0] // 7), num=4, dtype=int)

        elif r2:
            r1_x = np.linspace(int(ImageLayer.PADDING), int(ImageLayer.OUTPUT_SIZE[1] - ImageLayer.PADDING), num=r1, dtype=int)
            r2_x = np.linspace(int(ImageLayer.PADDING), int(ImageLayer.OUTPUT_SIZE[1] - ImageLayer.PADDING), num=r2, dtype=int)

            r1_y = np.linspace(int(2 * ImageLayer.OUTPUT_SIZE[0] // 5), int(1 * ImageLayer.OUTPUT_SIZE[0] // 5), num=4, dtype=int)
            r2_y = np.linspace(int(4 * ImageLayer.OUTPUT_SIZE[0] // 5), int(3 * ImageLayer.OUTPUT_SIZE[0] // 5), num=4, dtype=int)
        
        elif r1:
            r1_x = np.linspace(int(ImageLayer.PADDING), int(ImageLayer.OUTPUT_SIZE[1] - ImageLayer.PADDING), num=r1, dtype=int)

            r1_y = np.linspace(int(3 * ImageLayer.OUTPUT_SIZE[0] // 5), int(2 * ImageLayer.OUTPUT_SIZE[0] // 5), num=4, dtype=int)


        print('generating frames into buffer...')
        for frame_i,state in tqdm(enumerate(sol_path)):
            for bin_i in range(r1):
                for ball_i in range(4):
                    letter = state[4 * bin_i + ball_i]

                    if letter == ' ':
                        continue

                    color = ImageLayer.LETTER_TO_COLOR[letter]
                    pos = (r1_x[bin_i],r1_y[ball_i])
                    
                    cv2.circle(vid[frame_i], pos, ImageLayer.BALL_RAD, color, thickness=-1)

            for bin_i in range(r2):
                for ball_i in range(4):
                    letter = state[4 * (r1 + bin_i) + ball_i]

                    if letter == ' ':
                        continue

                    color = ImageLayer.LETTER_TO_COLOR[letter]
                    pos = (r2_x[bin_i],r2_y[ball_i])

                    cv2.circle(vid[frame_i], pos, ImageLayer.BALL_RAD, color, thickness=-1)

            for bin_i in range(r3):
                for ball_i in range(4):
                    letter = state[4 * (r1 + r2 + bin_i) + ball_i]

                    if letter == ' ':
                        continue

                    color = ImageLayer.LETTER_TO_COLOR[letter]
                    pos = (r3_x[bin_i],r3_y[ball_i])

                    cv2.circle(vid[frame_i], pos, ImageLayer.BALL_RAD, color, thickness=-1)
        
        writer = skvideo.io.FFmpegWriter(self.output_dir + filename + '.mp4', outputdict={
            '-vcodec': 'libx264',
            '-crf': '0',
            '-preset':'veryslow',
        })

        print('writing buffer to disk')
        for frame_i in tqdm(range(vid.shape[0])):
            writer.writeFrame(vid[frame_i,:,:,::-1])

        writer.close()
        cv2.destroyAllWindows()
