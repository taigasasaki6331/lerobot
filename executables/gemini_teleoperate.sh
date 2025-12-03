lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/follower_right \
    --robot.id=right_fol \
    --robot.cameras="{ front: {type: gemini, serial_number_or_name: 'CPBG152000D1', width: 1280, height: 720, fps: 30, use_depth: true}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/leader_right \
    --teleop.id=right_lea \
    --display_data=true
