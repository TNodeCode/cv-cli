from mm import save_faster_rcnn_pretrained

"""
save_faster_rcnn_pretrained(
    detector_config="mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    weights_file="work_dirs/faster_rcnn_50_frozen2/checkpoint0084.pth",
    filename_teacher="faster_rcnn_50_teacher_frozen2.pth",
    filename_student="faster_rcnn_50_student_frozen2.pth",
)
"""
save_faster_rcnn_pretrained(
    detector_config="mmdetection/mmdet/configs/deformable_detr/deformable_detr_r50_16xb2_50e_coco.py",
    weights_file="work_dirs/faster_rcnn_50_frozen2/checkpoint0084.pth",
    filename_teacher="deformble_detr_teacher_frozen2.pth",
    filename_student="deformable_detr_student_frozen2.pth",
)