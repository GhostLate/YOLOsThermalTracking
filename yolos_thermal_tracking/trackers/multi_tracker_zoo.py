from yolos_thermal_tracking.trackers.strong_sort.utils.parser import get_config
from yolos_thermal_tracking.trackers.strong_sort.strong_sort import StrongSORT
from yolos_thermal_tracking.trackers.ocsort.ocsort import OCSort
from yolos_thermal_tracking.trackers.bytetrack.byte_tracker import BYTETracker


def create_tracker(tracker_type, appearance_descriptor_weights=None, device='', half=False):
    if tracker_type == 'strongsort':
        assert appearance_descriptor_weights != None, 'ReID Weights arent given'
        cfg = get_config()
        cfg.merge_from_file('trackers/strong_sort/configs/strong_sort.yaml')

        strongsort = StrongSORT(
            appearance_descriptor_weights,
            device,
            half,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            max_unmatched_preds=cfg.STRONGSORT.MAX_UNMATCHED_PREDS,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

        )
        return strongsort
    elif tracker_type == 'ocsort':
        ocsort = OCSort(
            max_age=512,
            inertia=0.3,
            asso_func="ciou",
            det_thresh=0.45,
            iou_threshold=0.1,
            use_byte=False
        )
        return ocsort
    elif tracker_type == 'bytetrack':
        bytetracker = BYTETracker(
            track_thresh=0.75,
            track_buffer=512,
            match_thresh=0.9,
            frame_rate=30
        )
        return bytetracker
    else:
        print('No such tracker.py')
        exit()