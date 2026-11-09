import argparse
import json

IMAGE_GROUPS = {
    'vstar': ('vstar',),
    'hrbench_4k': ('hrbench_4k',),
    'hrbench_8k': ('hrbench_8k',),
    'visualprobe': ('visualprobe_easy', 'visualprobe_medium', 'visualprobe_hard'),
    'zoombench': ('zoombench',),
    'mme_realworld': ('mme_realworld',),
    'perceptionbench': ('perceptionbench',),
}
VIDEO_GROUPS = {name: (name,) for name in (
    'mvbench', 'videomme', 'videomme_v2', 'videoocr', 'vsi_bench',
    'mlvu', 'vstar_video',
)}
IMAGE_DATASETS = tuple(name for group in IMAGE_GROUPS.values() for name in group)
VIDEO_DATASETS = tuple(name for group in VIDEO_GROUPS.values() for name in group)
DATASETS = IMAGE_DATASETS + VIDEO_DATASETS


def jobs(modality='all', tool_mode='both'):
    names = DATASETS if modality == 'all' else IMAGE_DATASETS if modality == 'image' else VIDEO_DATASETS
    modes = ('w_tool', 'wo_tool') if tool_mode == 'both' else (tool_mode,)
    return [(name, mode, 'image' if name in IMAGE_DATASETS else 'video')
            for name in names for mode in modes]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=('all', 'image', 'video'), default='all')
    parser.add_argument('--tool-mode', choices=('both', 'w_tool', 'wo_tool'), default='both')
    parser.add_argument('--jobs', action='store_true', help='Print tab-separated executable jobs')
    args = parser.parse_args()
    if args.jobs:
        for row in jobs(args.modality, args.tool_mode):
            print('\t'.join(row))
    else:
        print(json.dumps({'image': IMAGE_GROUPS, 'video': VIDEO_GROUPS,
                          'visualprobe_aggregation': 'pool correct/total across easy, medium, hard; do not average split accuracies',
                          'jobs': jobs(args.modality, args.tool_mode)}, indent=2))
