"""
    @author: lntk
"""
from os.path import dirname, abspath
import sys

script_dir = dirname(dirname(dirname(abspath("x")))) + "/script"
sys.path.append(script_dir)

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.config import Config

Config.set('input', 'mouse', 'mouse, multitouch_on_demand')


class KaryotypingApp(App):
    def build(self):
        screen_manager = ScreenManager()
        from screen import build_screens
        return build_screens(screen_manager)


if __name__ == '__main__':
    KaryotypingApp().run()
