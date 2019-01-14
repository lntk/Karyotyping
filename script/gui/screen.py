"""
    @author: lntk
"""
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout

from script.gui.layout import DroppableLayout, ChromosomePairLayout
from script.gui.popup import ProgressBarPopup
from script.gui.image import ZoomableImage, DraggableImage, DroppableImage
from script.gui.button import CenteredButton, ButtonLayout

from script.util import general_utils, image_utils
from script.pipeline.pipeline import Pipeline

from os.path import dirname

data_dir = dirname(dirname(dirname(__file__))) + "/data/pipeline"


class ChromosomesScreen(Screen):
    def __init__(self, buttons, on_presses, **kw):
        super().__init__(**kw)

        """
        Initialize layout for chromosomes 
        """

        self.karyotyping_chromosomes_layout = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.85, 1.0))
        self.blocks = list()
        for i in range(7):
            row = BoxLayout(orientation='horizontal', spacing=10)
            for j in range(7):
                block = DraggableImage(source="")
                self.blocks.append(block)
                row.add_widget(block)

            self.karyotyping_chromosomes_layout.add_widget(row)

        self.button_layout = ButtonLayout(buttons, on_presses, size_hint=(0.15, 1.0))

        self.layout = BoxLayout(orientation='horizontal', spacing=10)
        self.layout.add_widget(self.karyotyping_chromosomes_layout)
        self.layout.add_widget(self.button_layout)

        self.add_widget(self.layout)

    def go_to_chromosome_cluster_screen(self, *args):
        self.manager.transition.direction = "right"
        self.manager.current = "generate_chromosome_cluster_screen"

    def go_to_raw_karyotyping_screen(self, *args):
        popup = ProgressBarPopup(auto_dismiss=False)
        popup.open()
        screen_manager = self.manager
        screen_manager = build_screens(screen_manager)

        screen_manager.transition.direction = "left"
        screen_manager.current = "raw_karyotyping_screen"
        popup.stop()

    def load_images(self, image_dir):
        pass


class ExtractChromosomesScreen(ChromosomesScreen):
    def __init__(self, **kw):
        """ SET UP LAYOUT """
        buttons = {
            "Straighten chromosomes": 0.2,
            "Back": 0.2,
            "empty space": 0.4,
            "Restart": 0.2
        }

        on_presses = {
            "Straighten chromosomes": self.start_straightening_chromosomes,
            "Back": self.go_to_chromosome_cluster_screen,
            "empty space": None,
            "Restart": self.go_to_raw_karyotyping_screen
        }

        super().__init__(buttons, on_presses, **kw)

        """ Set up properties """
        self.name = "extract_chromosomes_screen"
        self.image_dir_in = data_dir + "/extract_chromosomes"
        self.image_dir_out = data_dir + "/straighten_chromosomes"
        self.previous_screen = None
        self.input = None
        self.output = {
            "chromosomes": None
        }

        # """ Load images which are results from the previous process/screen """
        # self.load_images(self.image_dir_in)

    def post_init(self):
        self.previous_screen = self.manager.get_screen("generate_chromosome_cluster_screen")
        self.input = self.previous_screen.output

    def start_straightening_chromosomes(self, *args):
        """ Set up a 'loading' popup """
        popup = ProgressBarPopup()
        popup.open()
        popup.bind(on_dismiss=self.go_to_straighten_chromosomes_screen)

        """ Straighten chromosomes"""
        self.output["chromosomes"] = Pipeline.straighten_chromosomes(self.input["chromosomes"],
                                                                     save_dir=self.image_dir_out,
                                                                     popup=popup)
        """ Close popup after finishing processing """
        popup.stop()

    def go_to_straighten_chromosomes_screen(self, *args):
        """ Prepare next screen """
        screen = self.manager.get_screen("straighten_chromosomes_screen")
        screen.load_images(self.image_dir_out)

        self.manager.transition.direction = "left"
        self.manager.current = "straighten_chromosomes_screen"

    def load_images(self, image_dir):
        image_files = general_utils.get_all_files(image_dir)
        for image_file in image_files:
            idx = int(image_file[:-4]) - 1
            self.blocks[idx].reload_image(source=image_dir + "/" + image_file)


class StraightenChromosomesScreen(ChromosomesScreen):
    def __init__(self, **kw):
        """ SET UP LAYOUT """
        buttons = {
            "Detect interesting points": 0.2,
            "Back": 0.2,
            "empty space": 0.4,
            "Restart": 0.2
        }

        on_presses = {
            "Detect interesting points": self.start_detect_interesting_points_screen,
            "Back": self.go_to_extract_chromosomes_screen,
            "empty space": None,
            "Restart": self.go_to_raw_karyotyping_screen
        }

        super().__init__(buttons, on_presses, **kw)

        """ Set up properties """
        self.name = "straighten_chromosomes_screen"
        self.image_dir_in = data_dir + "/straighten_chromosomes"
        self.image_dir_out = data_dir + "/detect_interesting_points"
        self.previous_screen = None
        self.input = {
            "chromosomes": None
        }
        self.output = {
            "interesting_points": None
        }

        # """ Load images which are results from the previous process/screen """
        # self.load_images(self.image_dir_in)

    def post_init(self):
        self.previous_screen = self.manager.get_screen("extract_chromosomes_screen")
        self.input = self.previous_screen.output

    def start_detect_interesting_points_screen(self, *args):
        """ Set up a 'loading' popup """
        popup = ProgressBarPopup()
        popup.open()
        popup.bind(on_dismiss=self.go_to_detect_interesting_points_screen)

        """ Detect interesting points """
        self.output["interesting_points"] = Pipeline.not_detect_interesting_points(self.input["chromosomes"],
                                                                                   popup=popup)

        """ Close popup after finishing processing """
        popup.stop()

    def go_to_detect_interesting_points_screen(self, *args):
        """ Prepare next screen """
        screen = self.manager.get_screen("detect_interesting_points_screen")
        screen.load_images(self.image_dir_out)
        screen.input["chromosomes"] = self.input["chromosomes"]
        screen.input["points"] = self.output["interesting_points"]

        self.manager.transition.direction = "left"
        self.manager.current = "detect_interesting_points_screen"

    def go_to_extract_chromosomes_screen(self, *args):
        self.manager.transition.direction = "right"
        self.manager.current = "extract_chromosomes_screen"

    def load_images(self, image_dir):
        image_files = general_utils.get_all_files(image_dir)
        for image_file in image_files:
            parts = image_file[:-4].split("_")
            if len(parts) < 2:
                continue

            idx = int(parts[0])
            self.blocks[idx].reload_image(source=image_dir + "/" + image_file)


class DetectInterestingPointsScreen(ChromosomesScreen):
    def __init__(self, **kw):
        buttons = {
            "Classify chromosomes": 0.2,
            "Back": 0.2,
            "empty space": 0.4,
            "Restart": 0.2
        }

        on_presses = {
            "Classify chromosomes": self.start_classifying_chromosomes,
            "Back": self.go_to_straighten_chromosomes_screen,
            "empty space": None,
            "Restart": self.go_to_raw_karyotyping_screen
        }

        super().__init__(buttons, on_presses, **kw)

        """ Set up properties """
        self.name = "detect_interesting_points_screen"
        self.image_dir_in = data_dir + "/detect_interesting_points"
        self.image_dir_out = data_dir + "/classify_chromosomes"
        self.input = {
            "chromosomes": None,
            "points": None
        }
        self.output = {
            "classified_chromosomes": None
        }

        # """ Load images which are results from the previous process/screen """
        # self.load_images(self.image_dir_in)

    def start_classifying_chromosomes(self, *args):
        """ Set up a 'loading' popup """
        popup = ProgressBarPopup()
        popup.open()
        popup.bind(on_dismiss=self.go_to_classify_chromosomes_screen)

        """ Detect interesting points """
        self.output["classified_chromosomes"] = Pipeline.classify_chromosomes(self.input["chromosomes"],
                                                                              self.input["points"],
                                                                              save_dir=self.image_dir_out,
                                                                              popup=popup)

        """ Close popup after finishing processing """
        popup.stop()

    def go_to_classify_chromosomes_screen(self, *args):
        screen = self.manager.get_screen("karyotyping_screen")
        screen.load_images(self.image_dir_out)
        print(len(screen.karyotyping_chromosomes))
        print(len(screen.side_chromosomes))

        self.manager.transition.direction = "left"
        self.manager.current = "karyotyping_screen"

    def go_to_straighten_chromosomes_screen(self, *args):
        self.manager.transition.direction = "right"
        self.manager.current = "straighten_chromosomes_screen"

    def load_images(self, image_dir):
        image_files = general_utils.get_all_files(image_dir)
        for image_file in image_files:
            parts = image_file[:-4].split("_")
            if parts[1] == "draw":
                continue

            idx = int(parts[0]) - 1
            self.blocks[idx].reload_image(source=image_dir + "/" + image_file)


class RawKaryotypingScreen(Screen):
    def __init__(self, **kw):
        """ Set up properties """
        self.name = "raw_karyotyping_screen"
        self.image_dir_in = None
        self.image_dir_out = data_dir + "/generate_chromosome_cluster"

        self.input = {
            "image": None
        }
        self.output = {
            "chromosome_cluster": None
        }

        """ SET UP LAYOUT """
        super().__init__(**kw)
        buttons = {
            "Generate chromosome cluster": 0.2,
            "empty space": 0.6,
            "Restart": 0.2
        }

        on_presses = {
            "Generate chromosome cluster": self.start_generating_chromosome_cluster,
            "empty space": None,
            "Restart": self.test
        }

        self.button_layout = ButtonLayout(buttons, on_presses, size_hint=(0.15, 1.0))

        self.image_layout = AnchorLayout(size_hint=(0.85, 1.0), pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.image = ZoomableImage(source="", input_data=self.input, pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.image_layout.add_widget(self.image)

        self.layout = BoxLayout(orientation='horizontal', spacing=10)
        self.layout.add_widget(self.image_layout)
        self.layout.add_widget(self.button_layout)

        self.add_widget(self.layout)

    def start_generating_chromosome_cluster(self, *args):
        """ Set up a 'loading' popup """
        popup = ProgressBarPopup()
        popup.open()
        popup.bind(on_dismiss=self.go_to_generate_chromosome_cluster_screen)

        """ Generating chromosome cluster """
        self.output["chromosome_cluster"] = Pipeline.generate_chromosome_cluster(self.input["image"],
                                                                                 save_dir=self.image_dir_out,
                                                                                 popup=popup)
        """ Close popup after finishing processing """
        popup.stop()

    def go_to_generate_chromosome_cluster_screen(self, *args):
        screen = self.manager.get_screen("generate_chromosome_cluster_screen")
        screen.image.reload_image(self.image_dir_out + "/chromosome_cluster.bmp")

        self.manager.transition.direction = "left"
        self.manager.current = "generate_chromosome_cluster_screen"

    def test(self, *args):
        pass


class GenerateChromosomeClusterScreen(Screen):
    def __init__(self, **kw):
        """ SET UP LAYOUT """
        super().__init__(**kw)
        buttons = {
            "Extract chromosomes": 0.2,
            "Back": 0.2,
            "empty space": 0.4,
            "Restart": 0.2
        }

        on_presses = {
            "Extract chromosomes": self.start_extracting_chromosomes,
            "Back": self.go_to_raw_karyotyping_screen,
            "empty space": None,
            "Restart": self.go_to_raw_karyotyping_screen
        }

        self.button_layout = ButtonLayout(buttons, on_presses, size_hint=(0.15, 1.0))

        self.image_layout = AnchorLayout(size_hint=(0.85, 1.0))
        self.image = ZoomableImage(source="")
        self.image_layout.add_widget(self.image)

        self.layout = BoxLayout(orientation='horizontal', spacing=10)
        self.layout.add_widget(self.image_layout)
        self.layout.add_widget(self.button_layout)

        self.add_widget(self.layout)

        """ Set up properties """
        self.name = "generate_chromosome_cluster_screen"
        self.image_dir_in = data_dir + "/generate_chromosome_cluster"
        self.image_dir_out = data_dir + "/extract_chromosomes"
        self.previous_screen = None
        self.input = None
        self.output = {
            "chromosomes": None,
        }

    def post_init(self):
        self.previous_screen = self.manager.get_screen("raw_karyotyping_screen")
        self.input = self.previous_screen.output

    def start_extracting_chromosomes(self, *args):
        """ Set up a 'loading' popup """
        popup = ProgressBarPopup()
        popup.open()
        popup.bind(on_dismiss=self.go_to_extract_chromosomes_screen)

        """ Generating chromosome cluster """
        self.output["chromosomes"] = Pipeline.extract_chromosomes(self.input["chromosome_cluster"],
                                                                  save_dir=self.image_dir_out,
                                                                  popup=popup)

        """ Close popup after finishing processing """
        popup.stop()

    def go_to_extract_chromosomes_screen(self, *args):
        screen = self.manager.get_screen("extract_chromosomes_screen")
        screen.load_images(self.image_dir_out)

        self.manager.transition.direction = "left"
        self.manager.current = "extract_chromosomes_screen"

    def go_to_raw_karyotyping_screen(self, *args):
        popup = ProgressBarPopup(auto_dismiss=False)
        popup.open()
        screen_manager = self.manager

        # for screen in screen_manager.screens:
        #     del screen

        screen_manager = build_screens(screen_manager)

        screen_manager.transition.direction = "left"
        screen_manager.current = "raw_karyotyping_screen"
        popup.stop()

    def test(self, *args):
        pass


class KaryotypingScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "karyotyping_screen"
        self.karyotyping_chromosomes = list()
        self.side_chromosomes = list()
        self.image_dir_in = data_dir + "/classify_chromosomes"
        self.image_dir_out = None
        self.input = {
            "classified_chromosomes": None
        }
        self.output = {}

        """
        Layout for title
        """
        title_layout = BoxLayout(size_hint=(1.0, 0.1))
        self.karyotyping_button = Button(text="Restart")
        self.karyotyping_button.bind(on_press=self.go_to_raw_karyotyping_screen)
        title_layout.add_widget(self.karyotyping_button)

        """
        Layout for chromosomes
        """
        self.side_images_layout = DroppableLayout(orientation='vertical', size_hint=(0.1, 1.0), spacing=10)

        """
        Layout for karyotyping image
        """
        self.karyotyping_layout = self.get_karyotyping_layout(size_hint=(0.9, 1.0), spacing=5)

        """
        Layout for image for general
        """
        self.image_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 0.9), spacing=10)
        self.image_layout.add_widget(self.karyotyping_layout)
        self.image_layout.add_widget(self.side_images_layout)

        """
        Main layout
        """
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.layout.add_widget(title_layout)
        self.layout.add_widget(self.image_layout)

        """
        Add main layout to screen
        """
        self.add_widget(self.layout)

        """
        Update per drawing
        """
        Window.bind(on_touch_up=self.drop_chromosomes)

    def load_images(self, image_dir):
        num_chromosome = len(self.karyotyping_chromosomes)
        for idx in range(num_chromosome):
            chromosome = self.karyotyping_chromosomes[idx]
            source = image_dir + "/" + str(idx) + ".bmp"
            chromosome.reload_image(source)

    def go_to_raw_karyotyping_screen(self, *args):
        popup = ProgressBarPopup(auto_dismiss=False)
        popup.open()
        screen_manager = self.manager
        screen_manager = build_screens(screen_manager)

        screen_manager.transition.direction = "left"
        screen_manager.current = "raw_karyotyping_screen"
        popup.stop()

    def drop_chromosomes(self, window, touch):
        """ Check if we are in the current screen """
        if self.manager.current != self.name:
            return

        print(self.side_chromosomes)
        for chromosome in self.karyotyping_chromosomes:
            if self.side_images_layout.collide_point(*chromosome.container.center):
                # TODO: Make this less ugly
                chromosome_pair = chromosome.parent.parent

                chromosome_pair.remove_chromosome(chromosome)
                self.side_images_layout.add_image(chromosome)

                self.karyotyping_chromosomes.remove(chromosome)
                self.side_chromosomes.append(chromosome)

                # update chromosome's new position
                for side_chromosome in self.side_chromosomes:
                    side_chromosome.initial_pos = None
                break

        found = False
        for chromosome in self.side_chromosomes:
            for row in self.karyotyping_layout.children:
                for group in row.children:
                    for pair in group.children:
                        found = pair.is_addable(chromosome)
                        if found:
                            self.side_images_layout.remove_image(chromosome)
                            pair.add_chromosome(chromosome)

                            self.karyotyping_chromosomes.append(chromosome)
                            self.side_chromosomes.remove(chromosome)

                            # update chromosome's new position
                            for side_chromosome in self.side_chromosomes:
                                side_chromosome.initial_pos = None

                            # TODO: Find a way to early escape the loop
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break

    def check_mouse_inside_window(self, window_pos, window_size):
        x, y = self.mouse_pos
        x1, y1 = window_pos
        x2, y2 = x1 + window_size[0], y1 + window_size[1]

        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        else:
            return False

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
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_A_layout.add_widget(chromosome_pair_layout)
            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

        group_B_layout = BoxLayout(orientation='horizontal', size_hint=(0.4, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(3, 5):
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_B_layout.add_widget(chromosome_pair_layout)
            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

        row_1_layout.add_widget(group_A_layout)
        row_1_layout.add_widget(group_B_layout)

        """
        Layout for the second row, containing 1 cluster: (6, 7, ..., 12)
        """
        row_2_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_cluster_ratio)
        group_C_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(5, 12):
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_C_layout.add_widget(chromosome_pair_layout)
            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

        row_2_layout.add_widget(group_C_layout)

        """
        Layout for the third row, containing 2 clusters: (13, 14, 15) and (16, 17, 18)
        """
        row_3_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_cluster_ratio)
        group_D_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(12, 15):
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_D_layout.add_widget(chromosome_pair_layout)
            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

        group_E_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(15, 18):
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_E_layout.add_widget(chromosome_pair_layout)
            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

        row_3_layout.add_widget(group_D_layout)
        row_3_layout.add_widget(group_E_layout)

        """
        Layout for the fourth row, containing 3 clusters: (19, 20), (21, 22) and (x, y)
        """
        row_4_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_cluster_ratio)
        group_F_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(18, 20):
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_F_layout.add_widget(chromosome_pair_layout)
            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

        group_G_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(20, 22):
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_G_layout.add_widget(chromosome_pair_layout)
            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

        group_H_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 1.0), spacing=spacing * between_pair_ratio)
        for idx in range(22, 24):
            chromosome_pair_layout = ChromosomePairLayout(chromosome_ids[idx], spacing=spacing * between_chromosome_ratio)
            group_H_layout.add_widget(chromosome_pair_layout)

            # TODO: Modify this for xy-karyotyping image
            if idx == 23:
                continue

            self.karyotyping_chromosomes = self.karyotyping_chromosomes + chromosome_pair_layout.get_chromosomes()

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


class TestScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.left_chromosomes = list()
        self.right_chromosomes = list()

        self.left_layout = BoxLayout(orientation='vertical', spacing=10)
        self.chromosome = DraggableImage(source="")
        self.left_layout.add_widget(self.chromosome)
        self.left_chromosomes.append(self.chromosome)

        self.right_layout = DroppableLayout(orientation='vertical', spacing=10)

        self.layout = BoxLayout(orientation='horizontal', spacing=10, padding=10)
        self.layout.add_widget(self.left_layout)
        self.layout.add_widget(self.right_layout)

        Window.bind(on_touch_up=self.drop_chromosomes)

        self.add_widget(self.layout)

    def drop_chromosomes(self, *args):
        for chromosome in self.left_chromosomes:
            if self.right_layout.collide_point(*chromosome.center):
                print("Collided.")
                self.left_layout.remove_widget(chromosome)
                self.right_layout.add_widget(chromosome)

                self.left_chromosomes.remove(chromosome)
                self.right_chromosomes.append(chromosome)
                self.chromosome.initial_pos = None

                break


def build_screens(screen_manager):
    screen_manager.clear_widgets()

    screen_controller = {
        "raw_karyotyping_screen": True,
        "generate_chromosome_cluster_screen": True,
        "extract_chromosomes_screen": True,
        "straighten_chromosomes_screen": True,
        "detect_interesting_points_screen": True,
        "karyotyping_screen": True
    }

    if screen_controller["raw_karyotyping_screen"]:
        raw_karyotyping_screen = RawKaryotypingScreen()
        screen_manager.add_widget(raw_karyotyping_screen)

    if screen_controller["generate_chromosome_cluster_screen"]:
        generate_chromosome_cluster_screen = GenerateChromosomeClusterScreen()
        screen_manager.add_widget(generate_chromosome_cluster_screen)

        if screen_controller["raw_karyotyping_screen"]:
            generate_chromosome_cluster_screen.post_init()

    if screen_controller["extract_chromosomes_screen"]:
        extract_chromosomes_screen = ExtractChromosomesScreen()
        screen_manager.add_widget(extract_chromosomes_screen)

        if screen_controller["generate_chromosome_cluster_screen"]:
            extract_chromosomes_screen.post_init()

    if screen_controller["straighten_chromosomes_screen"]:
        straighten_chromosomes_screen = StraightenChromosomesScreen()
        screen_manager.add_widget(straighten_chromosomes_screen)

        if screen_controller["extract_chromosomes_screen"]:
            straighten_chromosomes_screen.post_init()

    if screen_controller["detect_interesting_points_screen"]:
        detect_interesting_points_screen = DetectInterestingPointsScreen()
        screen_manager.add_widget(detect_interesting_points_screen)

    if screen_controller["karyotyping_screen"]:
        karyotyping_screen = KaryotypingScreen()
        screen_manager.add_widget(karyotyping_screen)

    # screen_manager.add_widget(TestScreen())

    return screen_manager
