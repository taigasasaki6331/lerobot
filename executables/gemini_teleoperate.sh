lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/follower_right \
    --robot.id=right_fol \
    --robot.cameras="{ \
            top: {type: gemini, serial_number_or_name: 'CPBG152000D1', width: 1280, height: 720, fps: 30, use_depth: false}, \
            front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} \
        }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/leader_right \
    --teleop.id=right_lea \
    --display_data=true
