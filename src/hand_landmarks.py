import mediapipe as mp
import numpy as np
import torch

class HandLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # We need static_image_mode=True if we extract frame by frame independently
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_frame(self, frame):
        """
        Extracts hand landmarks from a single RGB frame.
        Args:
            frame: numpy array (H, W, 3) in RGB
        Returns:
            landmarks: numpy array of shape (42, 3) - 21 points per hand (x,y,z)
        """
        results = self.hands.process(frame)
        
        # Initialize with zeros for 2 hands, 21 landmarks, 3 coordinates
        landmarks = np.zeros((2, 21, 3), dtype=np.float32)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Ensure we only process up to 2 hands
                if hand_idx >= 2:
                    break
                
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                wrist_z = hand_landmarks.landmark[0].z
                
                for i, lm in enumerate(hand_landmarks.landmark):
                    landmarks[hand_idx, i, 0] = lm.x - wrist_x
                    landmarks[hand_idx, i, 1] = lm.y - wrist_y
                    landmarks[hand_idx, i, 2] = lm.z - wrist_z
                    
        # Flatten to (42*3,) or just return as (42, 3) -> flattening later
        return landmarks.reshape(-1)

    def extract_sequence(self, frames):
        """
        Extract landmarks from sequence of frames.
        Args:
            frames: torch Tensor (T, C, H, W)
        Returns:
            seq_landmarks: torch Tensor (T, 42*3)
        """
        # Convert to numpy (T, H, W, C), uint8
        frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        T = frames_np.shape[0]
        
        seq_lms = []
        for t in range(T):
            lms = self.extract_from_frame(frames_np[t])
            seq_lms.append(lms)
            
        return torch.from_numpy(np.array(seq_lms)).float()
