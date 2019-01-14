import kivy

kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from os.path import dirname, abspath
from functools import partial
import cv2

image_dir = dirname(dirname(abspath("X"))) + "/data/"


def parameter_tester(image_file, parameters, type='normal'):
    parameter_names = parameters.keys()
    temp_file = image_dir + 'temp.png'

    class MainScreen(Screen):
        def __init__(self, **kwargs):
            super(MainScreen, self).__init__(**kwargs)
            self.layout = BoxLayout(orientation='horizontal')
            num_param = len(parameter_names)

            multiple_sliders_layout = BoxLayout(orientation='vertical', size_hint=(0.3, 1.0))
            # pad empty box on top
            multiple_sliders_layout.add_widget(BoxLayout(orientation='vertical', size_hint=(1.0, 0.1 * (10 - num_param) / 2)))

            image = Image(source=image_file, size_hint=(0.7, 1.0))

            for name in parameter_names:
                single_slider_layout = BoxLayout(orientation='vertical', size_hint=(1.0, 0.1))
                name_part = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0))
                slider_name = Label(text=name)
                name_part.add_widget(slider_name)

                slider_part = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0))
                slider_value = Label(text='0.0', size_hint=(0.1, 1.0))
                slider_part.add_widget(slider_value)

                if type == 'threshold':
                    slider = Slider(min=parameters[name]['min_val'], max=parameters[name]['max_val'], size_hint=(0.9, 1.0), step=1)
                    slider.bind(value=partial(self.thresholding_update, slider_value, image))
                else:
                    slider = Slider(size_hint=(0.9, 1.0))
                    slider.bind(value=partial(self.normal_update, slider_value))

                slider_part.add_widget(slider)

                single_slider_layout.add_widget(name_part)
                single_slider_layout.add_widget(slider_part)
                multiple_sliders_layout.add_widget(single_slider_layout)

            # pad empty box at bottom
            multiple_sliders_layout.add_widget(BoxLayout(orientation='vertical', size_hint=(1.0, 0.1 * (10 - num_param) / 2)))
            self.layout.add_widget(multiple_sliders_layout)

            # temporarily save image
            cv2.imwrite(temp_file, cv2.imread(image_file))

            self.layout.add_widget(image)
            self.add_widget(self.layout)

        def normal_update(self, *args):
            slider_value, instance, value = args
            slider_value.text = '{:0.2f}'.format(value)

        def thresholding_update(self, *args):
            # get arguments
            slider_value, image, instance, value = args

            # threshold part
            gray = cv2.imread(image_file, 0)
            ret, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
            cv2.imwrite(temp_file, thresh)

            # update image
            image.source = temp_file
            image.reload()

            # update slider
            slider_value.text = '{:0.2f}'.format(value)

    screen_manager = ScreenManager()
    main_screen = MainScreen(name="main_screen")
    screen_manager.add_widget(main_screen)

    class ParameterTester(App):
        def build(self):
            return screen_manager

    sample_app = ParameterTester()
    sample_app.run()
