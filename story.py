import sys

from stories.four_warps import FourWarps


class Story:
    def __init__(self, name, run_fn, production_fn):
        self.name = name
        self.run_fn = run_fn
        self.production_fn = production_fn

    def run(self):
        print(f'Start running {self.name} story')
        self.run_fn()
        print(f'Finish running {self.name} story')

    def production(self):
        print(f'Start production of {self.name} story')
        self.production_fn()
        print(f'Finish production of {self.name} story')


STORIES = {'four_warps': FourWarps}


if __name__ == '__main__':
    story_name = sys.argv[1]
    assert story_name in STORIES.keys(), f'Story {story_name} not implemented'
    story = STORIES[story_name]()
    story.run()
    story.production()

