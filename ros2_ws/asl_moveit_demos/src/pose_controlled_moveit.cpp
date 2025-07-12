/**
 * @file pose_controlled_moveit.cpp
 * @brief A ROS 2 and MoveIt 2 node that subscribes to a target pose and moves the robot arm accordingly
 *
 * This program sets up a ROS 2 node that listens for target poses on a topic (e.g., "/target_pose")
 * and uses MoveIt 2 to plan and execute movements to reach those poses.
 *
 * @author Adapted by Copilot Space
 * @date July 11, 2025
 *
 * Based on hello_moveit.cpp:
 *
 * @author Addison Sears-Collins
 * @date December 15, 2024
 */

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>

class PoseControlledMoveit : public rclcpp::Node
{
public:
  PoseControlledMoveit()
  : Node("pose_controlled_moveit", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),
    arm_group_interface_(std::shared_ptr<rclcpp::Node>(this), "arm")
  {
    auto logger = this->get_logger();

    // MoveIt settings
    arm_group_interface_.setPlanningPipelineId("ompl");
    arm_group_interface_.setPlannerId("RRTConnectkConfigDefault");
    arm_group_interface_.setPlanningTime(1.0);
    arm_group_interface_.setMaxVelocityScalingFactor(1.0);
    arm_group_interface_.setMaxAccelerationScalingFactor(1.0);

    RCLCPP_INFO(logger, "Planning pipeline: %s", arm_group_interface_.getPlanningPipelineId().c_str());
    RCLCPP_INFO(logger, "Planner ID: %s", arm_group_interface_.getPlannerId().c_str());
    RCLCPP_INFO(logger, "Planning time: %.2f", arm_group_interface_.getPlanningTime());

    // Subscribe to target pose topic
    target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose",   // topic name
      10,              // QoS history depth
      std::bind(&PoseControlledMoveit::targetPoseCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(logger, "Ready to receive target poses on topic: /target_pose");
  }

private:
  void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    auto logger = this->get_logger();

    // Set the received pose as the target for the MoveGroupInterface
    arm_group_interface_.setPoseTarget(*msg);

    // Plan to that target pose
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(arm_group_interface_.plan(plan));

    if (success)
    {
      RCLCPP_INFO(logger, "Planning succeeded! Executing plan...");
      arm_group_interface_.execute(plan);
    }
    else
    {
      RCLCPP_ERROR(logger, "Planning failed!");
    }
  }

  moveit::planning_interface::MoveGroupInterface arm_group_interface_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PoseControlledMoveit>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
