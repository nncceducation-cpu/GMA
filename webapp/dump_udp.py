import inspect
from mmpose.structures.bbox import transforms as T
from mmpose.codecs import udp_heatmap as U
from mmpose.codecs.utils import post_processing as P
from mmpose.codecs.utils import refinement as R

print("### get_udp_warp_matrix ###")
print(inspect.getsource(T.get_udp_warp_matrix))
print("### bbox_xyxy2cs ###")
print(inspect.getsource(T.bbox_xyxy2cs))
print("### UDPHeatmap.decode ###")
print(inspect.getsource(U.UDPHeatmap.decode))
print("### get_heatmap_maximum ###")
print(inspect.getsource(P.get_heatmap_maximum))
print("### refine_keypoints_dark_udp ###")
print(inspect.getsource(R.refine_keypoints_dark_udp))
