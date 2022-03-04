from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils
import os
# from scipy.spatial import distance
# from dictances import bhattacharyya
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="image path")
args = vars(ap.parse_args())

class ClickPlot:
    """
    A clickable matplotlib figure

    Usage:
    # >>> import clickplot
    # >>> retval = clickplot.showClickPlot()
    # >>> print retval['subPlot']
    # >>> print retval['x']
    # >>> print retval['y']
    # >>> print retval['comment']

    See an example below
    """

    def __init__(self, fig=None):

        """
        Constructor

        Arguments:
        fig -- a matplotlib figure
        """

        if fig != None:
            self.fig = fig
        else:
            self.fig = plt.get_current_fig_manager().canvas.figure
        self.nSubPlots = len(self.fig.axes)
        self.dragFrom = None
        self.comment = 'Scroll up/down: rotate left/right; Arrow keys: strafe up/down/left/right; Press A/D: next/previous image'
        self.markers = []
        self.img1_o = img1
        self.img1 = img1
        self.img2 = img2
        self._img1 = img1
        self.id = 0
        self.stack_image = added_image

        #Init control options
        self.angle = 0

        self.retVal = {'comment': self.comment, 'x': None, 'y': None,
                       'subPlot': None}

        # self.sanityCheck()
        self.supTitle = plt.suptitle("comment: %s" % self.comment)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('button_release_event', self.onRelease)
        self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)

    def clearMarker(self):

        """Remove marker from retVal and plot"""

        self.retVal['x'] = None
        self.retVal['y'] = None
        self.retVal['subPlot'] = None
        for i in range(self.nSubPlots - 2):
            subPlot = self.selectSubPlot(i)
            for marker in self.markers:
                if marker in subPlot.lines:
                    subPlot.lines.remove(marker)
        self.markers = []
        self.fig.canvas.draw()

    def getSubPlotNr(self, event):

        """
        Get the nr of the subplot that has been clicked

        Arguments:
        event -- an event

        Returns:
        A number or None if no subplot has been clicked
        """

        i = 0
        axisNr = None
        for axis in self.fig.axes[:2]:
            if axis == event.inaxes:
                axisNr = i
                break
            i += 1
        return axisNr

    def sanityCheck(self):

        """Prints some warnings if the plot is not correct"""

        subPlot = self.selectSubPlot(0)
        minX = subPlot.dataLim.min[0]
        maxX = subPlot.dataLim.max[0]
        for i in range(self.nSubPlots):
            subPlot = self.selectSubPlot(i)
            _minX = subPlot.dataLim.min[0]
            _maxX = subPlot.dataLim.max[0]
            if abs(_minX - minX) != 0 or (_maxX - maxX) != 0:
                import warnings
                warnings.warn('Not all subplots have the same X-axis')

    def show(self):

        """
        Show the plot

        Returns:
        A dictionary with information about the response
        """

        plt.show()
        self.retVal['comment'] = self.comment
        return self.retVal

    def selectSubPlot(self, i):

        """
        Select a subplot

        Arguments:
        i -- the nr of the subplot to select

        Returns:
        A subplot
        """

        # plt.subplot(3,self.nSubPlots, i + 1)
        plt.subplot(2, 2, i + 1)
        return self.fig.axes[i]

    def reset_control(self):
        self.img1 = img1
        self.angle = 0
        for i in range(2):
            self.selectSubPlot(i).clear()

    def generate_content(self):
        self.clearMarker()



        self.stack_image = add_image(self.img1, self.img2)

        for i in range(self.nSubPlots):

            if i + 1 == 1:
                self.selectSubPlot(i).clear()
                self.markers.append([plt.imshow(self.img1), plt.title('2D image', fontsize=18)])

            if i + 1 == 2:
                self.selectSubPlot(i).clear()
                self.markers.append([plt.imshow(self.stack_image), plt.title('Stack transparent', fontsize=18)])

            if i + 1 == 4:
                self.selectSubPlot(i).clear()
                self.markers.append([plt.imshow(multiply_image(self.img1, self.img2)), plt.title('Multiply Image', fontsize=18)])

    def onClick(self, event):

        """
        Process a mouse click event. If a mouse is right clicked within a
        subplot, the return value is set to a (subPlotNr, xVal, yVal) tuple and
        the plot is closed. With right-clicking and dragging, the plot can be
        moved.

        Arguments:
        event -- a MouseEvent event
        """

        subPlotNr = self.getSubPlotNr(event)
        if subPlotNr == None:
            return

        if event.button == 1:
            # self.reset_control()
            self.generate_content()

            self.fig.canvas.draw()
            self.retVal['subPlot'] = subPlotNr
            self.retVal['x'] = event.xdata
            self.retVal['y'] = event.ydata

        else:
            # Start a dragFrom
            self.dragFrom = event.xdata

    def draw_onKey(self, comment):
        self.supTitle.set_text("comment: %s" % comment)
        self.fig.canvas.draw()

    def onKey(self, event):

        """
        Handle a keypress event. The plot is closed without return value on
        enter. Other keys are used to add a comment.

        Arguments:
        event -- a KeyEvent
        """
        # subPlotNr = self.getSubPlotNr(event)
        # if subPlotNr == None:
        #     return
        if event.key == 'q':
            plt.close()
            return

        if event.key == 'w':
            self.generate_content()
            self.draw_onKey('Auto aligning...')
            x_l, y_l, _ = optimize_local(self.img1, self.img2)
            self.img1 = shift_image(self.img1, x_l, y_l)

            self.generate_content()
            # self.draw_onKey('{}, {}'.format(np.mean(multiply_image(self.img1, self.img2)), optimize_local(self.img1, self.img2)))
            self.draw_onKey('Auto aligned!')
            return

        if event.key == 'up':
            self.img1 = shift_image(self.img1, 0, -5)

            self.generate_content()
            self.draw_onKey('shifted up by 1')

            return

        if event.key == 'down':
            self.img1 = shift_image(self.img1, 0, 5)

            self.generate_content()
            self.draw_onKey('shifted down by 1')
            return

        if event.key == 'right':
            self.img1 = shift_image(self.img1, 5, 0)

            self.generate_content()
            self.draw_onKey('shifted right by 1')
            return

        if event.key == 'left':
            self.img1 = shift_image(self.img1, -5, 0)

            self.generate_content()
            self.draw_onKey('shifted left by 1')
            return

        if event.key == 'z':
            self.angle += 10
            self.img1 = imutils.rotate(self.img1, 10)
            self.generate_content()

            self.fig.canvas.draw()
            # self.retVal['subPlot'] = subPlotNr
            return

        if event.key == 'v':
            self.angle -= 10
            self.img1 = imutils.rotate(self.img1, -10)
            self.generate_content()

            self.fig.canvas.draw()
            # self.retVal['subPlot'] = subPlotNr
            return

        if event.key == 'x':
            self.angle += 3
            self.img1 = imutils.rotate(self.img1, 1)
            self.generate_content()

            self.fig.canvas.draw()
            # self.retVal['subPlot'] = subPlotNr
            return

        if event.key == 'c':
            self.angle -= 3
            self.img1 = imutils.rotate(self.img1, -1)
            self.generate_content()

            self.fig.canvas.draw()
            # self.retVal['subPlot'] = subPlotNr
            return

        if event.key == 'backspace':
            self.clearMarker()
            self.reset_control()
            self.img1 = self.img1_o
            self.generate_content()
            self.draw_onKey('Reset to origin')
            return

        # if event.key == 't':
        #     self.selectSubPlot(3).clear()
        #     self.markers.append([plt.imshow(self.multiply_image(self.img1, self.img2)), plt.title('Multiply Image', fontsize=18)])
        #     self.draw_onKey('Reset to origin')
        #     return

        if event.key == 'd':
            self.clearMarker()
            self.reset_control()

            self.id += 1
            self.img1 = 255 - cv2.imread(paths[self.id], 0)[512:, :]
            self.img2 = 255 - cv2.imread(paths[self.id], 0)[:256, :]
            self.stack_image = add_image(self.img1, self.img2)
            self.img1_o = self.img1

            self.generate_content()
            self.selectSubPlot(2).clear()
            self.markers.append([plt.imshow(self.img2), plt.title('2D image', fontsize=18)])
            self.draw_onKey('Next Image')
            return

        if event.key == 'a':
            self.clearMarker()
            self.reset_control()

            self.id -= 1
            self.img1 = 255 - cv2.imread(paths[self.id], 0)[512:, :]
            self.img2 = 255 - cv2.imread(paths[self.id], 0)[:256, :]
            self.stack_image = add_image(self.img1, self.img2)
            self.img1_o = self.img1

            self.generate_content()
            self.selectSubPlot(2).clear()
            self.markers.append([plt.imshow(self.img2), plt.title('2D image', fontsize=18)])
            self.draw_onKey('previous Image')
            return

        # if event.key == 'backspace':
        #     self.comment = self.comment[:-1]
        elif len(event.key) == 1:
            self.comment += event.key
        self.supTitle.set_text("comment: %s" % self.comment)
        event.canvas.draw()

    def onRelease(self, event):

        """
        Handles a mouse release, which causes a move

        Arguments:
        event -- a mouse event
        """

        if self.dragFrom == None or event.button != 3:
            return
        dragTo = event.xdata
        dx = self.dragFrom - dragTo
        # for i in range(self.nSubPlots):
        for i in range(2):
            subPlot = self.selectSubPlot(i)
            xmin, xmax = subPlot.get_xlim()
            xmin += dx
            xmax += dx
            subPlot.set_xlim(xmin, xmax)
        event.canvas.draw()

    def onScroll(self, event):

        """
        Process scroll events. All subplots are scrolled simultaneously

        Arguments:
        event -- a MouseEvent
        """
        subPlotNr = self.getSubPlotNr(event)
        if subPlotNr == None:
            return
        # for i in range(self.nSubPlots):
        for i in range(2):
            subPlot = self.selectSubPlot(i)
            xmin, xmax = subPlot.get_xlim()
            dx = xmax - xmin
            cx = (xmax + xmin) / 2
            if event.button == 'down':
                dx *= 1.1
            else:
                dx /= 1.1
            _xmin = cx - dx / 2
            _xmax = cx + dx / 2
            subPlot.set_xlim(_xmin, _xmax)
        event.canvas.draw()

        # else:
        #     # Start a dragFrom
        #     self.dragFrom = self.x



def showClickPlot(fig=None):
    """
    Show a plt and return a dictionary with information

    Returns:
    A dictionary with the following keys:
    'subPlot' : the subplot or None if no marker has been set
    'x' : the X coordinate of the marker (or None)
    'y' : the Y coordinate of the marker (or None)
    'comment' : a comment string
    """

    cp = ClickPlot(fig)
    return cp.show()

def add_image(img1, img2):
    background = img1
    overlay = img2

    return cv2.addWeighted(background, 0.4, overlay, 0.1, 0)
def multiply_image(img1, img2):
    return cv2.bitwise_and(img1, img2)

def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def optimize_local(img1, img2):
    x_l, y_l, max_mean = 0, 0, 0
    for angle in range(-5,5):
        img1 = imutils.rotate(img1, angle)
        for x in range(-10, 10):
            for y in range(-10, 10):
                _img1 = shift_image(img1, x, y)
                img_mean = np.mean(multiply_image(_img1, img2))
                # print(img_mean)
                if max_mean < img_mean:
                    max_mean = img_mean
                    x_l = x
                    y_l = y
    return x_l, y_l, max_mean


def absoluteFilePaths(directory):
    list = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if '.png' in f:
                list.append(os.path.abspath(os.path.join(dirpath, f)))
    return list

if __name__ == '__main__':
    img_path = args["src"]

    if '.png' in img_path:
        img2 = 255 - cv2.imread(img_path, 0)[:256, :]
        img1 = 255 - cv2.imread(img_path, 0)[512:, :]

        added_image = add_image(img1, img2)

        fig = plt.figure(figsize=(16, 9))
        subPlot1 = fig.add_subplot(2,2,1)
        plt.imshow(img1), plt.title('2D', fontsize=18)

        subPlot2 = fig.add_subplot(2,2,2)
        plt.imshow(added_image), plt.title('Stack Transparent', fontsize=18)

        subPlot3 = fig.add_subplot(2,2,3)
        plt.imshow(img2), plt.title('Depthmap', fontsize=18)

        subPlot4 = fig.add_subplot(2,2,4)
        plt.imshow(multiply_image(img1, img2)), plt.title('Multiply Image', fontsize=18)

        plt.annotate('Scroll up/down: rotate left/right; Arrow keys: strafe up/down/left/right',
                     xy=(0.5, 0), xytext=(0, 10),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=14, ha='center', va='bottom')
        # Show the clickplot and print the return values
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        retval = showClickPlot()
        # print
        # 'Comment = %s' % retval['comment']
        # if retval['subPlot'] == None:
        #     print
        #     'No subplot selected'
        # else:
        #     print
        #     'You clicked in subplot %(subPlot)d at (%(x).3f, %(y).3f)' \
        #     % retval
    else:
        paths = absoluteFilePaths(img_path)
        img2 = 255 - cv2.imread(paths[0], 0)[:256, :]
        img1 = 255 - cv2.imread(paths[0], 0)[512:, :]

        added_image = add_image(img1, img2)
        fig = plt.figure(figsize=(16, 9))
        subPlot1 = fig.add_subplot(2, 2, 1)
        plt.imshow(img1), plt.title('2D', fontsize=18)

        subPlot2 = fig.add_subplot(2, 2, 2)

        plt.imshow(added_image), plt.title('Stack Transparent', fontsize=18)

        subPlot3 = fig.add_subplot(2, 2, 3)
        plt.imshow(img2), plt.title('Depthmap', fontsize=18)

        subPlot4 = fig.add_subplot(2, 2, 4)
        plt.imshow(multiply_image(img1, img2)), plt.title('Multiply Image', fontsize=18)
        # Show the clickplot and print the return values
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        retval = showClickPlot()
