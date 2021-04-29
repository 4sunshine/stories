import sys

from stories.four_warps import FourWarps


STORIES = {'four_warps': FourWarps}


if __name__ == '__main__':
    story_name = sys.argv[1]
    assert story_name in STORIES.keys(), f'Story {story_name} not implemented'
    story = STORIES[story_name]()
    story.run()
    story.production()

