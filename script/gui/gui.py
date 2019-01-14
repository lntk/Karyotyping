# Kivy example for the Popup widget

from kivy.app import App

from kivy.uix.button import Button

from kivy.uix.label import Label

from kivy.uix.popup import Popup

from kivy.uix.gridlayout import GridLayout


# Make an app by deriving from the kivy provided app class

class PopupExample(App):

    # override the build method and return the root widget of this App

    def build(self):
        # Define a grid layout for this App

        self.layout = GridLayout(cols=1, padding=10)

        # Add a button

        self.button = Button(text="Click for pop-up")

        self.layout.add_widget(self.button)

        # Attach a callback for the button press event

        self.button.bind(on_press=self.onButtonPress)

        return self.layout

    # On button press - Create a popup dialog with a label and a close button

    def onButtonPress(self, button):
        layout = GridLayout(cols=1, padding=10)

        popupLabel = Label(text="Click for pop-up")

        closeButton = Button(text="Close the pop-up")

        layout.add_widget(popupLabel)

        layout.add_widget(closeButton)

        # Instantiate the modal popup and display

        popup = Popup(title='Demo Popup',

                      content=layout)

        # content=(Label(text='This is a demo pop-up')))

        popup.open()

        # Attach close button press with popup.dismiss action

        closeButton.bind(on_press=popup.dismiss)

    # Run the app


if __name__ == '__main__':
    PopupExample().run()

# import cv2
# import numpy as np
# from os.path import dirname, abspath
# from gui.kivy_gui import parameter_tester
#
# image_dir = dirname(dirname(abspath("X"))) + "/data/"
# gray = cv2.imread(image_dir + "giemsa_raw.BMP", 0)
#
#
# def thresholding_gui(image_file):
#     parameters = {'threshold value': {'min_val': 0, 'max_val': 255}}
#     parameter_tester(image_file, parameters, type='threshold')
#
#
# def parameter_tester_gui(image_file, parameters):
#     from .kivy_gui import parameter_tester
#     parameter_tester(image_file, parameters)
#
#
# def threshold_gui(gray):
#     def nothing(x):
#         pass
#
#     # Create a black image, a window
#     img = np.zeros((300, 512, 3), np.uint8)
#     cv2.namedWindow('image')
#
#     # create trackbars for color change
#     cv2.createTrackbar('value', 'image', 0, 255, nothing)
#     threshold_value = 125
#
#     while True:
#         ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
#         cv2.imshow('image', thresh)
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27:
#             break
#
#         # get current positions of four trackbars
#         threshold_value = cv2.getTrackbarPos('value', 'image')
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     thresholding_gui("..../data/chromosome/1/xx_karyotype_001_0.bmp")
