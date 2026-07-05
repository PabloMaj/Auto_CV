"""
Hand-written user_prompt examples for evaluating DataAnalyserAgent's
`determine_desired_output` classification step.

20 examples per category, covering diverse CV domains (agriculture,
industrial inspection, medical, traffic/robotics, wildlife, ...).
Some prompts explicitly name the desired output ("... return bounding boxes"),
others only describe the task and require the model to infer the format,
so the set exercises both easy and hard classification cases.
"""

EXAMPLES = {
    "points": [
        "Detect the center point of each tomato in this greenhouse image.",
        "I need to locate the pupil center in each eye image for gaze tracking.",
        "Find the tip coordinates of each screw head on the PCB image.",
        "Identify the midpoint of every parking cone in the lot.",
        "Locate facial landmarks (eyes, nose, mouth corners) in portrait photos.",
        "Mark the center of each nail head in this carpentry photo.",
        "Detect keypoints on a human skeleton for pose estimation.",
        "Find the exact location of each bolt hole center on the metal plate.",
        "I want the coordinates of every bird's beak tip in these wildlife photos.",
        "Locate the centroid of each cell nucleus in the microscopy image.",
        "Detect the tip of each plant seedling shoot.",
        "Find the center coordinates of each traffic cone for robot navigation.",
        "Identify the corner points of a chessboard for camera calibration.",
        "Locate the center of each strawberry in the field image.",
        "Detect the apex point of each mountain peak silhouette in the photo.",
        "Mark the position of each drone in the swarm formation image.",
        "Find the geometric center of every button on the control panel.",
        "Locate each golf ball's position on the green.",
        "Detect the tip coordinates of surgical instruments in the endoscopy video frame.",
        "I need the center coordinate of every seed on the tray, not bounding boxes."
    ],
    "line_segments": [
        "Detect the lane markings on this highway image.",
        "Find the crop row lines in this UAV field image.",
        "Detect each power line cable segment in the image.",
        "Identify the edges of cracks in this concrete wall photo.",
        "Detect the conveyor belt guide lines in the factory image.",
        "Find the skeleton lines of the river network in this satellite image.",
        "Detect the boundary lines between adjacent farm plots.",
        "Identify wire segments in this circuit board photo.",
        "Detect the horizon line and any structural edges in the image.",
        "Find the centerline of each road in the aerial photo.",
        "Detect the seams between tiles on the floor image.",
        "Identify each fence line segment in the pasture photo.",
        "Detect the crease lines on this folded paper image.",
        "Find each pipeline segment in this industrial site image.",
        "Detect the boundary between the runway and grass in the airport image.",
        "Identify the veins (skeletonized) of a leaf in this scan.",
        "Detect each railway track segment in the image.",
        "Find the dividing lines of parking spaces in this lot photo.",
        "Detect cable segments running between utility poles.",
        "I need line segments representing each irrigation channel, not points."
    ],
    "bounding_boxes": [
        "Detect all cars in this parking lot image and return their locations.",
        "Find every apple on the tree in this orchard photo.",
        "Detect people in this crowd surveillance image.",
        "Identify all defective solder joints on the PCB image.",
        "Localize each weed in this crop field photo.",
        "Detect all traffic signs in this street view image.",
        "Find every box on the warehouse shelf in this image.",
        "Detect all fish in this underwater camera frame.",
        "Localize each tumor region in this medical scan.",
        "Detect all helmets worn by construction workers in this site photo.",
        "Find every cow in this pasture drone image.",
        "Detect all packages on the delivery truck image.",
        "Localize each pothole in this road surface photo.",
        "Detect all birds in this sky photo for counting.",
        "Find every ripe strawberry in this field image, marking its extent.",
        "Detect all forklifts operating in this warehouse camera feed.",
        "Localize each solar panel in this aerial rooftop image.",
        "Detect all life vests worn by passengers in this ferry image.",
        "Find every crate stacked in this storage yard image.",
        "I want a rectangle around each plant in the greenhouse image."
    ],
}
