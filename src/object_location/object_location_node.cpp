#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_2_algorithms.h>
#include "scene_graph/DetectedObjects.h"
#include "scene_graph/GraphObjects.h"


std::map<int, std::string> mseg_map = {
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


ros::Publisher debug_image_pub;
ros::Publisher debug_depth_pub;
ros::Publisher debug_odom_pub;
ros::Publisher graph_objects_pub;

geometry_msgs::Point32& convertToGlobalPoint(const nav_msgs::OdometryConstPtr& pose, geometry_msgs::Point32& t_point)
{
    geometry_msgs::Point32 local_point{t_point};

    // TODO: this offset is needed for the dataset - remove if needed!
    // Define a 90 degree rotation quaternion around the z-axis
    // tf2::Quaternion q_0;
    // q_0.setRPY(0, M_PI_2, 0);  // Roll = 0, Pitch = 0, Yaw = 90 degrees (M_PI/2 radians)

    // // // Convert the quaternion to a rotation matrix
    // tf2::Matrix3x3 rotation_matrix_0(q_0);

    // float x = t_point.x;
    // float y = t_point.y;
    // float z = t_point.z;

    // // Rotate the point around the z-axis
    // local_point.x = rotation_matrix_0[0][0] * x + rotation_matrix_0[0][1] * y + rotation_matrix_0[0][2] * z;
    // local_point.y = rotation_matrix_0[1][0] * x + rotation_matrix_0[1][1] * y + rotation_matrix_0[1][2] * z;
    // local_point.z = rotation_matrix_0[2][0] * x + rotation_matrix_0[2][1] * y + rotation_matrix_0[2][2] * z;

    // --- end of offset calculation

    geometry_msgs::Point32 global_point{};

    tf2::Quaternion q{pose->pose.pose.orientation.x, pose->pose.pose.orientation.y, pose->pose.pose.orientation.z, pose->pose.pose.orientation.w};
    tf2::Matrix3x3 rotation_matrix{q};

    // Rotate the local point by the robot's orientation
    tf2::Vector3 local_point_tf(local_point.x, local_point.y, local_point.z);
    tf2::Vector3 rotated_point = rotation_matrix * local_point_tf;

    // Translate the rotated point by the robot's global position
    
    global_point.x = rotated_point.x() + pose->pose.pose.position.x;
    global_point.y = rotated_point.y() + pose->pose.pose.position.y;
    global_point.z = rotated_point.z() + pose->pose.pose.position.z;

    t_point = global_point;

    return t_point;
}


geometry_msgs::Point32& rotatePoint(geometry_msgs::Point32& t_point, double roll, double pitch, double yaw)
{
    // geometry_msgs::Point32 local_point{t_point};

    // TODO: this offset is needed for the dataset - remove if needed!
    // Define a 90 degree rotation quaternion around the z-axis
    tf2::Quaternion q_0;
    q_0.setRPY(roll, pitch, yaw);  // Roll = 0, Pitch = 0, Yaw = 90 degrees (M_PI/2 radians)

    // // Convert the quaternion to a rotation matrix
    tf2::Matrix3x3 rotation_matrix_0(q_0);

    float x = t_point.x;
    float y = t_point.y;
    float z = t_point.z;

    // Rotate the point around the z-axis
    t_point.x = rotation_matrix_0[0][0] * x + rotation_matrix_0[0][1] * y + rotation_matrix_0[0][2] * z;
    t_point.y = rotation_matrix_0[1][0] * x + rotation_matrix_0[1][1] * y + rotation_matrix_0[1][2] * z;
    t_point.z = rotation_matrix_0[2][0] * x + rotation_matrix_0[2][1] * y + rotation_matrix_0[2][2] * z;

    return t_point;
}


std::string getClassNameFromID(int t_id)
{
    if (mseg_map.find(t_id) != mseg_map.end()) {
        return mseg_map[t_id];
    } else {
        return "";
    }
}


float computeMedian(std::vector<float>& values)
{
    size_t n = values.size();
    
    if (n == 0) {
        throw std::invalid_argument("Cannot compute median of an empty vector");
    }

    std::sort(values.begin(), values.end());

    if (n % 2 == 1) {
        // Odd number of elements
        return values[n / 2];
    } else {
        // Even number of elements
        return (values[(n / 2) - 1] + values[n / 2]) / 2.0;
    }
}


geometry_msgs::Point32 computeMedianPoint(const std::vector<geometry_msgs::Point32>& points)
{
    if (points.empty()) {
        throw std::invalid_argument("The input vector of points is empty");
    }

    std::vector<float> x_values, y_values, z_values;

    // Extract the x, y, and z values from points
    for (const auto& point : points) {
        x_values.push_back(point.x);
        y_values.push_back(point.y);
        z_values.push_back(point.z);
    }

    // Compute the median for each coordinate
    float median_x = computeMedian(x_values);
    float median_y = computeMedian(y_values);
    float median_z = computeMedian(z_values);

    // Create and return the median point
    geometry_msgs::Point32 median_point;
    median_point.x = median_x;
    median_point.y = median_y;
    median_point.z = median_z;

    return median_point;
}


double calculateDistance(const geometry_msgs::Point32& p1, const geometry_msgs::Point32& p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}


bool isPointInPolygon(const std::vector<geometry_msgs::Point32>& t_polygon, float t_px, float t_py)
{
       // Define the kernel
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

    // Define the types for points and polygon
    typedef K::Point_2 Point;
    typedef CGAL::Polygon_2<K> Polygon;

    // Define a polygon
    Polygon polygon{};

    for (auto& point : t_polygon)
        polygon.push_back(Point(point.x, point.y));

    Point point{t_px, t_py};

    // Check if the point is inside the polygon
    CGAL::Bounded_side result = CGAL::bounded_side_2(polygon.vertices_begin(), polygon.vertices_end(), point, K());

    return result == CGAL::ON_BOUNDED_SIDE || result == CGAL::ON_BOUNDARY ? true : false;
}



std::vector<geometry_msgs::Point32> getNearestPoints(const std::vector<geometry_msgs::Point32>& points, const geometry_msgs::Point32& center)
{
     // Create a vector to store distances along with corresponding points
    std::vector<std::pair<double, geometry_msgs::Point32>> distances;

    // Calculate the distance of each point from the center and store it
    for (const auto& point : points) {
        double distance = calculateDistance(point, center);
        distances.push_back(std::make_pair(distance, point));
    }

    // Sort the distances vector based on the distances
    std::sort(distances.begin(), distances.end(), [](const std::pair<double, geometry_msgs::Point32>& a, const std::pair<double, geometry_msgs::Point32>& b) {
        return a.first < b.first;
    });

    // Get the number of points that represent the closest 75%
    int num_points_to_select = static_cast<int>(distances.size() * 0.70);

    // Extract the closest 50% points
    std::vector<geometry_msgs::Point32> nearest_points;
    for (size_t i = 0; i < num_points_to_select; ++i) {
        nearest_points.push_back(distances[i].second);
    }

    return nearest_points;
}


std::pair<geometry_msgs::Point32, geometry_msgs::Point32> computeBoundingBox(const std::vector<geometry_msgs::Point32>& points)
{
    if (points.empty()) {
        throw std::invalid_argument("The input vector of points is empty");
    }

    // Initialize min and max points with the first point in the vector
    geometry_msgs::Point32 min_point = points[0];
    geometry_msgs::Point32 max_point = points[0];

    // Iterate through all points to find the min and max for each coordinate
    for (const auto& point : points) {
        min_point.x = std::min(min_point.x, point.x);
        min_point.y = std::min(min_point.y, point.y);
        min_point.z = std::min(min_point.z, point.z);

        max_point.x = std::max(max_point.x, point.x);
        max_point.y = std::max(max_point.y, point.y);
        max_point.z = std::max(max_point.z, point.z);
    }

    // Return a pair containing the min and max points of the bounding box
    return std::make_pair(min_point, max_point);
}


void synchronized_callback(const sensor_msgs::ImageConstPtr& image,
              const sensor_msgs::PointCloud2ConstPtr& depth_image,
              const nav_msgs::OdometryConstPtr& pose,
              const scene_graph::DetectedObjectsConstPtr& detected_objects)
{
    // Process synchronized messages
    ROS_INFO("Received synchronized messages");

    sensor_msgs::PointCloud2 cloud_out;
    cloud_out = *depth_image;  // Copy the header and other metadata

    // Use PointCloud2 iterators to modify the data in the cloud
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_out, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_out, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_out, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {

        geometry_msgs::Point32 p{};
        p.x = *iter_x;
        p.y = *iter_y - 0.65;
        p.z = *iter_z;

        p = rotatePoint(p, 0, 0, -M_PI_2);
        p = rotatePoint(p, 0, M_PI_2, 0);

        p = convertToGlobalPoint(pose, p);

        geometry_msgs::Point32 global_point{};

        *iter_x = p.x;
        *iter_y = p.y;
        *iter_z = p.z;
    }

    debug_image_pub.publish(*image);
    debug_depth_pub.publish(cloud_out);
    debug_odom_pub.publish(*pose);


    // if (m_depth_image == nullptr || m_camera_pose == nullptr || m_camera_image == nullptr)
    //     return;

    // ros::Time start_time = ros::Time::now();

    scene_graph::GraphObjects graph_objects{};
    graph_objects.header.stamp = ros::Time::now();

    int i = 0;
    for (const auto& detected_object : detected_objects->objects) {

        std::cout << "detected object: " << detected_object.class_name << std::endl;

        std::vector<geometry_msgs::Point32> pointcloud_segment{};

        // std::chrono::steady_clock::time_point before_loop = std::chrono::steady_clock::now();

        auto delta_y{static_cast<int>(detected_object.bounding_box.at(1).y) - static_cast<int>(detected_object.bounding_box.at(0).y)};
        auto delta_x{static_cast<int>(detected_object.bounding_box.at(1).x) - static_cast<int>(detected_object.bounding_box.at(0).x)};

        for (auto y = static_cast<int>(detected_object.bounding_box.at(0).y); y < static_cast<int>(detected_object.bounding_box.at(1).y); ++y) {
            for (auto x = static_cast<int>(detected_object.bounding_box.at(0).x); x < static_cast<int>(detected_object.bounding_box.at(1).x); ++x) {

                if (!isPointInPolygon(detected_object.segment, x, y))
                    continue;

                int array_position{static_cast<int>(y * image->width + x)};

                sensor_msgs::PointCloud2ConstIterator<float> iter_x{cloud_out, "x"};
                sensor_msgs::PointCloud2ConstIterator<float> iter_y{cloud_out, "y"};
                sensor_msgs::PointCloud2ConstIterator<float> iter_z{cloud_out, "z"};

                iter_x += array_position;
                iter_y += array_position;
                iter_z += array_position;

                if (*iter_z > 0.0) {
                    geometry_msgs::Point32 p{};
                    p.x = *iter_x;
                    p.y = *iter_y;
                    p.z = *iter_z;

                    pointcloud_segment.push_back(p);
                }
            }
        }


        if (pointcloud_segment.size() == 0)
            continue;

        auto center_point{computeMedianPoint(pointcloud_segment)};
        auto filtered_pointcloud_segment{getNearestPoints(pointcloud_segment, center_point)};

        if (filtered_pointcloud_segment.size() == 0)
            continue;

        // for (auto& point : filtered_pointcloud_segment)
        //     point = convertToGlobalPoint(pose, point);

        auto bounding_box{computeBoundingBox(filtered_pointcloud_segment)};

        scene_graph::GraphObject graph_object{};
        graph_object.name = detected_object.class_name;
        graph_object.bounding_box.push_back(bounding_box.first);
        graph_object.bounding_box.push_back(bounding_box.second);

        graph_objects.objects.push_back(graph_object);

        ++i;
    }


    graph_objects_pub.publish(graph_objects);

    // // LOG
    // ros::Time end_time = ros::Time::now();

    // // Calculate the elapsed time
    // ros::Duration elapsed_time = end_time - start_time;

    // std::ofstream file{};
    // file.open("/home/nes/.ros/log/object_location_node.log", std::ios::app);

    //     if (file.is_open()) {
    //     file << "[TIMING]: " << std::fixed << std::setprecision(6) << elapsed_time.toSec() << std::endl;

    //     // Close the file
    //     file.close();
    // } else {
    //     std::cerr << "Unable to open the file." << std::endl;
    // }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "synchronizer_node");
    ros::NodeHandle nh;

    debug_image_pub = nh.advertise<sensor_msgs::Image>("/scene_graph/debug/image", 10);
    debug_depth_pub = nh.advertise<sensor_msgs::PointCloud2>("/scene_graph/debug/depth", 10);
    debug_odom_pub = nh.advertise<nav_msgs::Odometry>("/scene_graph/debug/odom", 10);
    graph_objects_pub = nh.advertise<scene_graph::GraphObjects>("/scene_graph/seen_graph_objects", 100);

    // Subscribers to the topics
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/scene_graph/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud_sub(nh, "/scene_graph/depth/points", 1);
    message_filters::Subscriber<nav_msgs::Odometry> pose_sub(nh, "/scene_graph/odom", 1);
    message_filters::Subscriber<scene_graph::DetectedObjects> detected_objects_sub(nh, "/scene_graph/detected_objects", 1);

    // Approximate Time Synchronizer Policy (useful if timestamps are not perfectly synchronized)
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2, nav_msgs::Odometry, scene_graph::DetectedObjects> MySyncPolicy;

    // Synchronizer
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, pointcloud_sub, pose_sub, detected_objects_sub);
    sync.registerCallback(boost::bind(&synchronized_callback, _1, _2, _3, _4));

    ros::spin();
    return 0;
}
