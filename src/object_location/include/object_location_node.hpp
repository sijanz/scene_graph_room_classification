#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <map>

class ObjectLocationModule
{
public:
    ObjectLocationModule();

    void detectedObjectsCallback(const scene_graph::DetectedObjectsConstPtr& msg);
    void detectedObjectsMsegCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void cameraPoseCallback(const nav_msgs::OdometryConstPtr& msg);
    void cameraImageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
    ros::NodeHandle m_nh;
    
    ros::Subscriber m_detected_objects_sub;
    ros::Subscriber m_depth_sub;
    ros::Subscriber m_camera_pose_sub;
    ros::Subscriber m_camera_image_sub;
    ros::Subscriber m_detected_objects_mseg_sub;

    ros::Publisher m_graph_objects_pub;
    ros::Publisher m_seen_object_center_pub;
    ros::Publisher m_pointcloud_pub;
    ros::Publisher m_segment_pointcloud_pub;

    sensor_msgs::PointCloud2ConstPtr m_depth_image;
    nav_msgs::OdometryConstPtr m_camera_pose;
    sensor_msgs::ImageConstPtr m_camera_image;
    bool m_debug_mode;

    std::string getClassNameFromID(int id);
    geometry_msgs::Point32& convertToGlobalPoint(geometry_msgs::Point32& point);
    double yawFromPose(const geometry_msgs::Pose& pose);
    visualization_msgs::Marker createMarkerFromPoint(const geometry_msgs::Point32& point, int i) const;
    std::pair<geometry_msgs::Point32, geometry_msgs::Point32> computeBoundingBox(const std::vector<geometry_msgs::Point32>& points) const;
    float computeMedian(std::vector<float>& values) const;
    geometry_msgs::Point32 computeMedianPoint(const std::vector<geometry_msgs::Point32>& points) const;
    sensor_msgs::PointCloud2 createPointCloud2FromPoints(const std::vector<geometry_msgs::Point32>& points) const;
    bool isPointOnLine(int x, int y, int x1, int y1, int x2, int y2) const;
    int orientation(int px, int py, int qx, int qy, int rx, int ry) const;
    bool doIntersect(int p1x, int p1y, int q1x, int q1y, int p2x, int p2y, int q2x, int q2y) const;
    bool isPointInPolygon(const std::vector<geometry_msgs::Point32>& polygon, float px, float py) const;
    double calculateDistance(const geometry_msgs::Point32& p1, const geometry_msgs::Point32& p2) const;
    std::vector<geometry_msgs::Point32> getNearestPoints(const std::vector<geometry_msgs::Point32>& points, const geometry_msgs::Point32& center) const;

    std::map<int, std::string> m_mseg_map = {
        {0, "backpack"}, {1, "umbrella"}, {2, "bag"}, {3, "tie"}, {4, "suitcase"},
        {5, "case"}, {6, "bird"}, {7, "cat"}, {8, "dog"}, {9, "horse"},
        {10, "sheep"}, {11, "cow"}, {12, "elephant"}, {13, "bear"}, {14, "zebra"},
        {15, "giraffe"}, {16, "animal_other"}, {17, "microwave"}, {18, "radiator"}, {19, "oven"},
        {20, "toaster"}, {21, "storage_tank"}, {22, "conveyor_belt"}, {23, "sink"}, {24, "refrigerator"},
        {25, "washer_dryer"}, {26, "fan"}, {27, "dishwasher"}, {28, "toilet"}, {29, "bathtub"},
        {30, "shower"}, {31, "tunnel"}, {32, "bridge"}, {33, "pier_wharf"}, {34, "tent"},
        {35, "building"}, {36, "ceiling"}, {37, "laptop"}, {38, "keyboard"}, {39, "mouse"},
        {40, "remote"}, {41, "cell phone"}, {42, "television"}, {43, "floor"}, {44, "stage"},
        {45, "banana"}, {46, "apple"}, {47, "sandwich"}, {48, "orange"}, {49, "broccoli"},
        {50, "carrot"}, {51, "hot_dog"}, {52, "pizza"}, {53, "donut"}, {54, "cake"},
        {55, "fruit_other"}, {56, "food_other"}, {57, "chair_other"}, {58, "armchair"}, {59, "swivel_chair"},
        {60, "stool"}, {61, "seat"}, {62, "couch"}, {63, "trash_can"}, {64, "potted_plant"},
        {65, "nightstand"}, {66, "bed"}, {67, "table"}, {68, "pool_table"}, {69, "barrel"},
        {70, "desk"}, {71, "ottoman"}, {72, "wardrobe"}, {73, "crib"}, {74, "basket"},
        {75, "chest_of_drawers"}, {76, "bookshelf"}, {77, "counter_other"}, {78, "bathroom_counter"}, {79, "kitchen_island"},
        {80, "door"}, {81, "light_other"}, {82, "lamp"}, {83, "sconce"}, {84, "chandelier"},
        {85, "mirror"}, {86, "whiteboard"}, {87, "shelf"}, {88, "stairs"}, {89, "escalator"},
        {90, "cabinet"}, {91, "fireplace"}, {92, "stove"}, {93, "arcade_machine"}, {94, "gravel"},
        {95, "platform"}, {96, "playingfield"}, {97, "railroad"}, {98, "road"}, {99, "snow"},
        {100, "sidewalk_pavement"}, {101, "runway"}, {102, "terrain"}, {103, "book"}, {104, "box"},
        {105, "clock"}, {106, "vase"}, {107, "scissors"}, {108, "plaything_other"}, {109, "teddy_bear"},
        {110, "hair_dryer"}, {111, "toothbrush"}, {112, "painting"}, {113, "poster"}, {114, "bulletin_board"},
        {115, "bottle"}, {116, "cup"}, {117, "wine_glass"}, {118, "knife"}, {119, "fork"},
        {120, "spoon"}, {121, "bowl"}, {122, "tray"}, {123, "range_hood"}, {124, "plate"},
        {125, "person"}, {126, "rider_other"}, {127, "bicyclist"}, {128, "motorcyclist"}, {129, "paper"},
        {130, "streetlight"}, {131, "road_barrier"}, {132, "mailbox"}, {133, "cctv_camera"}, {134, "junction_box"},
        {135, "traffic_sign"}, {136, "traffic_light"}, {137, "fire_hydrant"}, {138, "parking_meter"}, {139, "bench"},
        {140, "bike_rack"}, {141, "billboard"}, {142, "sky"}, {143, "pole"}, {144, "fence"},
        {145, "railing_banister"}, {146, "guard_rail"}, {147, "mountain_hill"}, {148, "rock"}, {149, "frisbee"},
        {150, "skis"}, {151, "snowboard"}, {152, "sports_ball"}, {153, "kite"}, {154, "baseball_bat"},
        {155, "baseball_glove"}, {156, "skateboard"}, {157, "surfboard"}, {158, "tennis_racket"}, {159, "net"},
        {160, "base"}, {161, "sculpture"}, {162, "column"}, {163, "fountain"}, {164, "awning"},
        {165, "apparel"}, {166, "banner"}, {167, "flag"}, {168, "blanket"}, {169, "curtain_other"},
        {170, "shower_curtain"}, {171, "pillow"}, {172, "towel"}, {173, "rug_floormat"}, {174, "vegetation"},
        {175, "bicycle"}, {176, "car"}, {177, "autorickshaw"}, {178, "motorcycle"}, {179, "airplane"},
        {180, "bus"}, {181, "train"}, {182, "truck"}, {183, "trailer"}, {184, "boat_ship"},
        {185, "slow_wheeled_object"}, {186, "river_lake"}, {187, "sea"}, {188, "water_other"}, {189, "swimming_pool"},
        {190, "waterfall"}, {191, "wall"}, {192, "window"}, {193, "window_blind"}
    };
};
