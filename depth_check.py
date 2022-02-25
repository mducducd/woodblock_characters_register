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
        self.img1 = img1
        self.img2 = img2
        self.id = 0

        #Init control options
        self. strafe_x = 0
        self.strafe_y = 0
        self.x = 0
        self.y = 0
        self.angle = 0

        self.retVal = {'comment': self.comment, 'x': None, 'y': None,
                       'subPlot': None}

        self.sanityCheck()
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
        for i in range(self.nSubPlots):
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
        plt.subplot(3, 2, i + 1)
        return self.fig.axes[i]

    def reset_control(self):
        self.img1 = img1
        self.img2 = img2
        self. strafe_x = 0
        self.strafe_y = 0
        self.x = 0
        self.y = 0
        self.angle = 0
        for i in range(self.nSubPlots):
            self.selectSubPlot(i).clear()

    def bhattacharyya(self, h1, h2):
        '''Calculates the Byattacharyya distance of two histograms.'''

        def normalize(h):
            return h / np.sum(h)

        return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))

    def generate_content(self, x, y, angle=None, strafe_x =0, strafe_y = 0):
        self.clearMarker()
        if angle is not None:
            self.img1 = imutils.rotate(img1, angle)
            self.img2 = imutils.rotate(img2, angle)
        # else:
        #     self.img1 = img1
        #     self.img2 = img2
        list_bin_y_img1 = [self.img1[i, y] for i in range(img1.shape[0])]
        list_bin_y_img2 = [self.img2[i, y] for i in range(img2.shape[0])]
        list_bin_x_img1 = [self.img1[x, i] for i in range(img1.shape[1])]
        list_bin_x_img2 = [self.img2[x, i] for i in range(img2.shape[1])]

        dist1 = self.bhattacharyya(np.array(list_bin_y_img1), np.array(list_bin_y_img2))
        dist2 = self.bhattacharyya(np.array(list_bin_x_img1), np.array(list_bin_x_img2))

        for i in range(self.nSubPlots):
            subPlot = self.selectSubPlot(i)
            if i + 1 == 1:
                self.selectSubPlot(i).clear()
                # marker = plt.axvline(event.xdata, 0, 1, linestyle='--', \
                #                      linewidth=2, color='b')
                # marker = plt.axvline(x, 0, 1, linewidth=2, color='g')
                # # marker = plt.scatter(event.xdata, event.ydata, s=50, color='r')
                # self.markers.append(marker)
                # marker = plt.axhline(y, 0, 1, linewidth=2, color='b')
                # self.markers.append(marker)
                # marker = plt.scatter(x, y, s=100, color='r')
                self.markers.append([plt.imshow(self.img1, cmap='gray'), plt.title('Real Depthmap', fontsize=24), \
                                     plt.axvline(x, 0, 1, linewidth=2, color='b', zorder=1),\
                                     plt.axhline(y, 0, 1, linewidth=2, color='g', zorder=1),\
                                     plt.scatter(x, y, s=100, color='r', zorder=2)])

            if i + 1 == 2:
                self.selectSubPlot(i).clear()
                self.markers.append([plt.imshow(self.img2, cmap='gray'), plt.title('Fake Depthmap', fontsize=24), \
                                     plt.axvline(x, 0, 1, linewidth=2, color='b', zorder=1),\
                                     plt.axhline(y, 0, 1, linewidth=2, color='g', zorder=1),\
                                     plt.scatter(x, y, s=100, color='r', zorder=2)])


            if i + 1 == 3:
                self.selectSubPlot(i).clear()
                # marker = [plt.plot(list_bin_y_img1, color='g', zorder=1), \
                #             plt.scatter(x, img1[x, y], s=50, color='r', zorder=2), \
                #             plt.ylim(0, 300)]
                self.markers.append([plt.plot(list_bin_y_img1, color='g', zorder=1), \
                                     plt.scatter(x, self.img1[x, y], s=100, color='r', zorder=2), \
                                     plt.ylim(0, 300)])

            if i + 1 == 4:
                self.selectSubPlot(i).clear()
                # marker = [plt.plot(list_bin_y_img2, color='g', zorder=1), \
                #           plt.scatter(x, img2[x, y], s=50, color='r', zorder=2), \
                #           plt.ylim(0, 300)]
                self.markers.append([plt.plot(list_bin_y_img2, color='g', zorder=1), \
                                     plt.scatter(x, self.img2[x, y], s=100, color='r', zorder=2), \
                                     plt.ylim(0, 300), plt.text(310, 150, np.round(dist1, 5), size=24, ha="center")])
            if i + 1 == 5:
                self.selectSubPlot(i).clear()
                # marker = [plt.plot(list_bin_x_img1, color='b', zorder=1), \
                #           plt.scatter(y, img1[x,y], s=50, color='r', zorder=2), \
                #           plt.ylim(0, 300)]
                self.markers.append([plt.plot(list_bin_x_img1, color='b', zorder=1), \
                                     plt.scatter(y, self.img1[x, y], s=100, color='r', zorder=2), \
                                     plt.ylim(0, 300)])
            if i + 1 == 6:
                self.selectSubPlot(i).clear()
                # marker = [plt.plot(list_bin_x_img2, color='b', zorder=1), \
                #           plt.scatter(y, img2[x,y], s=50, color='r', zorder=2), \
                #           plt.ylim(0, 300)]
                self.markers.append([plt.plot(list_bin_x_img2, color='b', zorder=1), \
                                     plt.scatter(y, self.img2[x, y], s=100, color='r', zorder=2), \
                                     plt.ylim(0, 300), plt.text(310, 150, np.round(dist2, 5), size=24, ha="center"), \
                                     plt.annotate('Scroll up/down: rotate left/right; Arrow keys: strafe up/down/left/right; Press A/D: next/previous image',
                                                  xy=(0.5, 0), xytext=(0, 10),
                                                  xycoords=('axes fraction', 'figure fraction'),
                                                  textcoords='offset points',
                                                  size=14, ha='center', va='bottom')
                                     ])

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
            self.x = int(event.xdata)
            self.y = int(event.ydata)
            self.generate_content(self.x, self.y)

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

        if event.key == 'q':
            plt.close()
            return

        if event.key == 'up':
            self.y -= 5
            self.generate_content(self.x, self.y)
            self.draw_onKey('y = y + 1')
            return

        if event.key == 'down':
            self.y += 5
            self.generate_content(self.x, self.y)
            self.draw_onKey('y = y - 1')
            return

        if event.key == 'left':
            self.x -= 5
            self.generate_content(self.x, self.y)
            self.draw_onKey('x = x - 1')
            return

        if event.key == 'right':
            self.x += 5
            self.generate_content(self.x, self.y)
            self.draw_onKey('x = x + 1')
            return

        if event.key == 'backspace':
            self.clearMarker()
            self.reset_control()
            self.generate_content(self.x, self.y)
            self.draw_onKey('Reset to origin')
            return

        if event.key == 'd':
            self.clearMarker()
            self.reset_control()

            self.id += 1
            self.img1 = pad_img(255 - cv2.imread(paths[self.id], 0)[512:, :])
            self.img2 = pad_img(255 - cv2.imread(paths[self.id], 0)[256:512, :])
            self.generate_content(self.x, self.y)
            self.draw_onKey('Next Image')
            return

        if event.key == 'a':
            self.clearMarker()
            self.reset_control()

            self.id -= 1
            self.img1 = pad_img(255 - cv2.imread(paths[self.id], 0)[512:, :])
            self.img2 = pad_img(255 - cv2.imread(paths[self.id], 0)[256:512, :])
            self.generate_content(self.x, self.y)
            self.draw_onKey('Next Image')
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
        # for i in range(2):
        #     subPlot = self.selectSubPlot(i)
        #     xmin, xmax = subPlot.get_xlim()
        #     dx = xmax - xmin
        #     cx = (xmax + xmin) / 2
        #     if event.button == 'down':
        #         dx *= 1.1
        #     else:
        #         dx /= 1.1
        #     _xmin = cx - dx / 2
        #     _xmax = cx + dx / 2
        #     subPlot.set_xlim(_xmin, _xmax)
        # event.canvas.draw()
        if event.button == 'down':
            self.angle += 30
            self.generate_content(self.x, self.y, self.angle)

            self.fig.canvas.draw()
            self.retVal['subPlot'] = subPlotNr
            self.retVal['x'] = self.x
            self.retVal['y'] = self.y
        if event.button == 'up':
            self.angle -= 30
            self.generate_content(self.x, self.y, self.angle)

            self.fig.canvas.draw()
            self.retVal['subPlot'] = subPlotNr
            self.retVal['x'] = self.x
            self.retVal['y'] = self.y
        else:
            # Start a dragFrom
            self.dragFrom = self.x



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

def pad_img(img):
    pad_value = imutils.rotate_bound(img, 45).shape[0] - img.shape[0]
    # padding 20px to print_image mask
    h ,w = img.shape
    img = cv2.resize(img, (h-pad_value,w-pad_value)) ######################
    ht, wd = img.shape
    # result = np.full((hh,ww), color, dtype=np.uint8)
    thresh2 = np.zeros((h, w))
    # compute center offset
    xx = (h - wd) // 2
    yy = (w - ht) // 2
    # copy img image into center of result image
    thresh2[yy:yy+ht, xx:xx+wd] = img

    return thresh2

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
        img1 = pad_img(255 - cv2.imread(img_path, 0)[512:, :])
        img2 = pad_img(255 - cv2.imread(img_path, 0)[256:512, :])

        fig = plt.figure(figsize=(16, 9))
        subPlot1 = fig.add_subplot(3,2,1)
        plt.imshow(img1, cmap='gray'), plt.title('Real Depthmap', fontsize=24)
        # plt.plot(xData, yData1, figure=fig)
        subPlot2 = fig.add_subplot(3,2,2)
        plt.imshow(img2, cmap='gray'), plt.title('Fake Depthmap', fontsize=24)
        # plt.plot(xData, yData2, figure=fig)
        subPlot3 = fig.add_subplot(3,2,3)
        # plt.imshow(img2, cmap='gray')
        subPlot4 = fig.add_subplot(3,2,4)
        # plt.imshow(img2, cmap='gray')
        subPlot5 = fig.add_subplot(3,2,5)
        # plt.imshow(img2, cmap='gray')
        subPlot6 = fig.add_subplot(3,2,6)
        # plt.imshow(img2, cmap='gray')
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
        img1 = pad_img(255 - cv2.imread(paths[0], 0)[512:, :])
        img2 = pad_img(255 - cv2.imread(paths[0], 0)[256:512, :])

        fig = plt.figure(figsize=(16, 9))
        subPlot1 = fig.add_subplot(3, 2, 1)
        plt.imshow(img1, cmap='gray'), plt.title('Real Depthmap', fontsize=24)
        # plt.plot(xData, yData1, figure=fig)
        subPlot2 = fig.add_subplot(3, 2, 2)
        plt.imshow(img2, cmap='gray'), plt.title('Fake Depthmap', fontsize=24)
        # plt.plot(xData, yData2, figure=fig)
        subPlot3 = fig.add_subplot(3, 2, 3)
        # plt.imshow(img2, cmap='gray')
        subPlot4 = fig.add_subplot(3, 2, 4)
        # plt.imshow(img2, cmap='gray')
        subPlot5 = fig.add_subplot(3, 2, 5)
        # plt.imshow(img2, cmap='gray')
        subPlot6 = fig.add_subplot(3, 2, 6)
        # plt.imshow(img2, cmap='gray')
        plt.annotate('Scroll up/down: rotate left/right; Arrow keys: strafe up/down/left/right',
                     xy=(0.5, 0), xytext=(0, 10),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=14, ha='center', va='bottom')
        # Show the clickplot and print the return values
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        retval = showClickPlot()