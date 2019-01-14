"""
    @author: lntk
"""

import kivy
# kivy.require("1.11.0")
from kivy.app import App
from kivy.graphics.transformation import Matrix
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from os.path import dirname, abspath
from functools import partial
import cv2
from glob import glob
from random import randint
from os.path import join, dirname

from kivy.properties import StringProperty, Logger

from kivy.config import Config

from .image import DraggableImage

Window.clearcolor = (0.5, 0.5, 0.5, 1)
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')


class UpdatableBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_widget(self, widget, index=0, canvas=None):
        super(UpdatableBoxLayout, self).add_widget(widget=widget, index=index, canvas=canvas)
        if type(widget) is DraggableImage:
            widget.reload()


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)

        """
        Layout for title
        """
        title_layout = BoxLayout(size_hint=(1.0, 0.1))
        title_layout.add_widget(Button(text="Drag and drop"))

        """
        Layout for mouse events
        """
        mouse_layout = BoxLayout(size_hint=(1.0, 0.1))
        self.mouse_label = Label()
        self.mouse_pos = tuple()
        mouse_layout.add_widget(self.mouse_label)
        Window.bind(mouse_pos=lambda w, p: setattr(self.mouse_label, 'text', str(p)))
        Window.bind(mouse_pos=lambda w, p: setattr(self, 'mouse_pos', p))

        """
        Layout for chromosomes
        """
        self.side_images_layout = BoxLayout(orientation='vertical', size_hint=(0.1, 1.0), spacing=10)
        image = DraggableImage(source='/home/lntk/Desktop/Karyotype/script/null/gui/pictures/images/Wall.jpg')
        self.side_images_layout.add_widget(image)
        Window.bind(on_dropfile=self._on_file_drop)

        """
        Layout for karyotyping image
        """
        self.karyotyping_layout = self.get_karyotyping_layout(size_hint=(0.9, 1.0), spacing=5)

        """
        Layout for image for general
        """
        self.image_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=10)
        self.image_layout.add_widget(self.karyotyping_layout)
        self.image_layout.add_widget(self.side_images_layout)

        """
        Main layout
        """
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.layout.add_widget(title_layout)
        self.layout.add_widget(mouse_layout)
        self.layout.add_widget(self.image_layout)

        """
        Add main layout to screen
        """
        self.add_widget(self.layout)

    def on_pause(self):
        return True

    def _on_file_drop(self, window, file_path):
        for row in self.karyotyping_layout.children:
            for group in row.children:
                for chromosome__pair in group.children:
                    pair = chromosome__pair.children[1]
                    for image in pair.children:
                        if self.check_mouse_inside_window(image.pos, image.size):
                            filePath = file_path.decode("utf-8")
                            print(file_path)
                            image.source = filePath
                            image.reload()
                            return

    def check_mouse_inside_window(self, window_pos, window_size):
        x, y = self.mouse_pos
        x1, y1 = window_pos
        x2, y2 = x1 + window_size[0], y1 + window_size[1]

        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        else:
            return False

    def get_chromosome_block_layout(self, name, size_hint=(1.0, 1.0), spacing=10):
        chromosome_pairs_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 0.9), spacing=spacing)
        chromosome_1 = Image(source="", size_hint=(1.0, 1.0))
        chromosome_2 = Image(source="", size_hint=(1.0, 1.0))
        chromosome_pairs_layout.add_widget(chromosome_1)
        chromosome_pairs_layout.add_widget(chromosome_2)

        name_layout = BoxLayout(orientation='vertical', size_hint=(1.0, 0.1), spacing=spacing)
        name_layout.add_widget(Label(text=name))

        layout = BoxLayout(orientation='vertical', size_hint=size_hint, spacing=spacing)
        layout.add_widget(chromosome_pairs_layout)
        layout.add_widget(name_layout)

        return layout

    def get_karyotyping_layout(self, size_hint, spacing=5):
        between_cluster_ratio = 16
        between_pair_ratio = 2
        between_chromosome_ratio = 1
        between_row_ratio = 4

        chromosome_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "x", "y"]

        """
        Layout for the first row, containing 2 clusters: (1, 2, 3) and (4, 5)
        """
        row_1_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_cluster_ratio)
        group_A_layout = BoxLayout(orientation='horizontal', size_hint=(0.6, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(0, 3):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_A_layout.add_widget(chromosome__pair_layout)

        group_B_layout = BoxLayout(orientation='horizontal', size_hint=(0.4, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(3, 5):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_B_layout.add_widget(chromosome__pair_layout)

        row_1_layout.add_widget(group_A_layout)
        row_1_layout.add_widget(group_B_layout)

        """
        Layout for the second row, containing 1 cluster: (6, 7, ..., 12)
        """
        row_2_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_cluster_ratio)
        group_C_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(5, 12):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_C_layout.add_widget(chromosome__pair_layout)

        row_2_layout.add_widget(group_C_layout)

        """
        Layout for the third row, containing 2 clusters: (13, 14, 15) and (16, 17, 18)
        """
        row_3_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_cluster_ratio)
        group_D_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(12, 15):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_D_layout.add_widget(chromosome__pair_layout)

        group_E_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(15, 18):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_E_layout.add_widget(chromosome__pair_layout)

        row_3_layout.add_widget(group_D_layout)
        row_3_layout.add_widget(group_E_layout)

        """
        Layout for the fourth row, containing 3 clusters: (19, 20), (21, 22) and (x, y)
        """
        row_4_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_cluster_ratio)
        group_F_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(18, 20):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_F_layout.add_widget(chromosome__pair_layout)

        group_G_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(20, 22):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_G_layout.add_widget(chromosome__pair_layout)

        group_H_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(22, 24):
            chromosome__pair_layout = self.get_chromosome_block_layout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_H_layout.add_widget(chromosome__pair_layout)

        row_4_layout.add_widget(group_F_layout)
        row_4_layout.add_widget(group_G_layout)
        row_4_layout.add_widget(group_H_layout)

        """
        Main layout
        """
        karyotyping_layout = BoxLayout(orientation='vertical', size_hint=size_hint, spacing=spacing * between_row_ratio)
        karyotyping_layout.add_widget(row_1_layout)
        karyotyping_layout.add_widget(row_2_layout)
        karyotyping_layout.add_widget(row_3_layout)
        karyotyping_layout.add_widget(row_4_layout)

        return karyotyping_layout


class DragDropApp(App):
    def build(self):
        screen_manager = ScreenManager()
        main_screen = MainScreen(name="main_screen")
        screen_manager.add_widget(main_screen)
        return screen_manager


if __name__ == "__main__":
    DragDropApp().run()
