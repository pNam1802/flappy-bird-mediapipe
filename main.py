import cv2
import mediapipe as mp
import pygame
import time
import random
import os
import numpy as np
from collections import deque

pygame.init()

# ============== MEDIAPIPE SETUP ==============
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# ============== PYGAME SETUP ==============
WIDTH, HEIGHT = 1000, 680          # Tăng kích thước màn hình cho đẹp hơn
GAME_WIDTH = 560                   # Vùng chơi rộng hơn
CAM_WIDTH = WIDTH - GAME_WIDTH - 60
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chim Bay - Điều khiển bằng tay")
clock = pygame.time.Clock()

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
SKY_BLUE = (135, 206, 235)
YELLOW = (255, 215, 0)
RED = (220, 20, 60)
ORANGE = (255, 165, 0)
GRAY = (240, 240, 240)

# ============== GAME PHYSICS ==============
bird_x = 120
bird_y = HEIGHT // 2
bird_vel = 0
GRAVITY = 0.4
FLAP_POWER = -9
MAX_FALL_SPEED = 8

pipe_gap = 250          # Tăng khoảng cách giữa ống để dễ chơi hơn
pipe_width = 60
pipe_speed = 5.5
pipes = []
PIPE_SPAWN_TIME = 3.5   # Ống xuất hiện thưa hơn

def add_pipe():
    min_height = 80
    max_height = HEIGHT - pipe_gap - 80
    height = random.randint(min_height, max_height)
    pipes.append([GAME_WIDTH, height])

# High score
HIGHSCORE_FILE = "highscore.txt"

def load_highscore():
    if os.path.exists(HIGHSCORE_FILE):
        try:
            with open(HIGHSCORE_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def save_highscore(score):
    with open(HIGHSCORE_FILE, 'w') as f:
        f.write(str(score))

high_score = load_highscore()

# ============== GESTURE DETECTION ==============
class GestureDetector:
    def __init__(self, smoothing_frames=5):
        self.smoothing_frames = smoothing_frames
        self.left_wrist_history = deque(maxlen=smoothing_frames)
        self.right_wrist_history = deque(maxlen=smoothing_frames)
        self.prev_left_y = None
        self.prev_right_y = None
        self.left_velocity_history = deque(maxlen=3)
        self.right_velocity_history = deque(maxlen=3)
        self.last_flap_time = 0
        self.FLAP_COOLDOWN = 0.25
        self.velocity_threshold = 15
        self.debug_info = ""
        self.flap_visual_timer = 0

    def smooth_value(self, history, new_value):
        history.append(new_value)
        if len(history) < 2:
            return new_value
        return sum(history) / len(history)

    def detect_flap(self, landmarks, frame_height):
        current_time = time.time()
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

        min_visibility = 0.6
        left_visible = left_wrist.visibility > min_visibility and left_elbow.visibility > min_visibility
        right_visible = right_wrist.visibility > min_visibility and right_elbow.visibility > min_visibility

        if not left_visible and not right_visible:
            self.debug_info = "Không thấy tay!"
            return False

        left_y = self.smooth_value(self.left_wrist_history, left_wrist.y * frame_height) if left_visible else None
        right_y = self.smooth_value(self.right_wrist_history, right_wrist.y * frame_height) if right_visible else None

        flap_detected = False
        left_vel = right_vel = 0

        if left_visible and self.prev_left_y is not None:
            left_vel = left_y - self.prev_left_y
            self.left_velocity_history.append(left_vel)

        if right_visible and self.prev_right_y is not None:
            right_vel = right_y - self.prev_right_y
            self.right_velocity_history.append(right_vel)

        if left_visible: self.prev_left_y = left_y
        if right_visible: self.prev_right_y = right_y

        avg_left_vel = sum(self.left_velocity_history) / len(self.left_velocity_history) if self.left_velocity_history else 0
        avg_right_vel = sum(self.right_velocity_history) / len(self.right_velocity_history) if self.right_velocity_history else 0

        time_since_last = current_time - self.last_flap_time
        can_flap = time_since_last > self.FLAP_COOLDOWN

        if can_flap and (avg_left_vel > self.velocity_threshold or avg_right_vel > self.velocity_threshold):
            flap_detected = True
            self.last_flap_time = current_time
            self.flap_visual_timer = current_time + 0.3
            self.left_velocity_history.clear()
            self.right_velocity_history.clear()

        self.debug_info = f"L:{avg_left_vel:.1f} R:{avg_right_vel:.1f}"
        if flap_detected:
            self.debug_info += " VÂY!"
        elif not can_flap:
            self.debug_info += f" CD:{self.FLAP_COOLDOWN - time_since_last:.1f}s"

        return flap_detected

    def is_flap_visual_active(self):
        return time.time() < self.flap_visual_timer

    def reset(self):
        self.left_wrist_history.clear()
        self.right_wrist_history.clear()
        self.left_velocity_history.clear()
        self.right_velocity_history.clear()
        self.prev_left_y = None
        self.prev_right_y = None

gesture_detector = GestureDetector(smoothing_frames=4)

# ============== GAME STATES ==============
WAITING = "WAITING"
COUNTDOWN = "COUNTDOWN"
PLAYING = "PLAYING"
GAMEOVER = "GAMEOVER"

game_state = WAITING
countdown_start_time = 0
countdown_value = 3
fingers_hold_frames = 0
FINGERS_HOLD_REQUIRED = 20

running = True
score = 0
restart_delay = 0
RESTART_WAIT = 2.0
scored_pipes = set()
last_pipe = time.time()

# ============== WEBCAM ==============
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Fonts
title_font = pygame.font.SysFont("Arial", 48, bold=True)
score_font = pygame.font.SysFont("Arial", 48, bold=True)      # To hơn
small_font = pygame.font.SysFont("Arial", 32, bold=True)      # To hơn

# ============== MAIN LOOP ==============
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            if event.key == pygame.K_SPACE and game_state == PLAYING:
                bird_vel = FLAP_POWER

    # Webcam processing
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    frame_height = frame.shape[0]
    flap_detected = False
    fingers_count = 0

    # Detect fingers (for start)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            fingers_count = 0
            for tip_id in [8, 12, 16, 20]:
                if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                    fingers_count += 1
            if landmarks[4].x > landmarks[3].x:
                fingers_count += 1
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Pose processing for flap
    if pose_results.pose_landmarks and game_state == PLAYING:
        landmarks = pose_results.pose_landmarks.landmark
        flap_detected = gesture_detector.detect_flap(landmarks, frame_height)

        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        h, w = frame.shape[:2]
        color = (0, 255, 255) if gesture_detector.is_flap_visual_active() else (255, 255, 0)
        radius = 25 if gesture_detector.is_flap_visual_active() else 15

        for wrist in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
            lm = landmarks[wrist]
            if lm.visibility > 0.6:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius, color, -1)

        cv2.putText(frame, gesture_detector.debug_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ============== GAME STATE LOGIC ==============
    if game_state == WAITING:
        if fingers_count == 5:
            fingers_hold_frames += 1
            if fingers_hold_frames >= FINGERS_HOLD_REQUIRED:
                game_state = COUNTDOWN
                countdown_start_time = time.time()
                countdown_value = 3
                fingers_hold_frames = 0
        else:
            fingers_hold_frames = 0

    elif game_state == COUNTDOWN:
        elapsed = time.time() - countdown_start_time
        countdown_value = 3 - int(elapsed)
        if countdown_value <= 0:
            game_state = PLAYING
            bird_y = HEIGHT // 2
            bird_vel = 0
            pipes = []
            scored_pipes.clear()
            add_pipe()
            last_pipe = time.time()
            score = 0
            gesture_detector.reset()

    elif game_state == PLAYING:
        bird_vel += GRAVITY
        bird_vel = min(bird_vel, MAX_FALL_SPEED)
        bird_y += bird_vel

        if flap_detected:
            bird_vel = FLAP_POWER

        if time.time() - last_pipe > PIPE_SPAWN_TIME:
            add_pipe()
            last_pipe = time.time()

        for pipe in pipes[:]:
            pipe[0] -= pipe_speed
            pipe_id = id(pipe)
            if pipe[0] + pipe_width < bird_x and pipe_id not in scored_pipes:
                score += 1
                scored_pipes.add(pipe_id)
            if pipe[0] + pipe_width < 0:
                pipes.remove(pipe)
                scored_pipes.discard(pipe_id)

        bird_rect = pygame.Rect(bird_x - 20, bird_y - 20, 40, 40)
        collided = False
        for pipe in pipes:
            top_rect = pygame.Rect(pipe[0], 0, pipe_width, pipe[1])
            bottom_rect = pygame.Rect(pipe[0], pipe[1] + pipe_gap, pipe_width, HEIGHT - pipe[1] - pipe_gap)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                collided = True
                break

        if collided or bird_y < 20 or bird_y > HEIGHT - 20:
            game_state = GAMEOVER
            if score > high_score:
                high_score = score
                save_highscore(high_score)
            restart_delay = time.time() + RESTART_WAIT

    elif game_state == GAMEOVER:
        if time.time() > restart_delay:
            game_state = WAITING
            fingers_hold_frames = 0

    # ============== RENDERING ==============
    screen.fill(GRAY)

    # Vùng game
    game_rect = pygame.Rect(20, 20, GAME_WIDTH, HEIGHT - 40)
    pygame.draw.rect(screen, SKY_BLUE, game_rect)

    # Gradient sky
    for y in range(game_rect.height):
        color_val = 135 + int((y / game_rect.height) * 60)
        pygame.draw.line(screen, (color_val, 206, 235),
                         (game_rect.left, game_rect.top + y),
                         (game_rect.right, game_rect.top + y))

    # Ground
    pygame.draw.rect(screen, (139, 119, 101), (game_rect.left, HEIGHT - 70, GAME_WIDTH, 50))
    pygame.draw.rect(screen, (34, 139, 34), (game_rect.left, HEIGHT - 75, GAME_WIDTH, 10))

    # Pipes
    for pipe in pipes:
        x = pipe[0] + game_rect.left
        # Top
        pygame.draw.rect(screen, GREEN, (x, 0, pipe_width, pipe[1]))
        pygame.draw.rect(screen, DARK_GREEN, (x, 0, pipe_width, pipe[1]), 4)
        pygame.draw.rect(screen, GREEN, (x - 6, pipe[1] - 24, pipe_width + 12, 24))
        pygame.draw.rect(screen, DARK_GREEN, (x - 6, pipe[1] - 24, pipe_width + 12, 24), 4)
        # Bottom
        bottom_y = pipe[1] + pipe_gap
        pygame.draw.rect(screen, GREEN, (x, bottom_y, pipe_width, HEIGHT - bottom_y))
        pygame.draw.rect(screen, DARK_GREEN, (x, bottom_y, pipe_width, HEIGHT - bottom_y), 4)
        pygame.draw.rect(screen, GREEN, (x - 6, bottom_y, pipe_width + 12, 24))
        pygame.draw.rect(screen, DARK_GREEN, (x - 6, bottom_y, pipe_width + 12, 24), 4)

    # Bird
    if game_state == PLAYING:
        bird_color = ORANGE if flap_detected or gesture_detector.is_flap_visual_active() else YELLOW
        pygame.draw.circle(screen, bird_color, (bird_x + game_rect.left, int(bird_y)), 24)  # To hơn
        pygame.draw.circle(screen, BLACK, (bird_x + game_rect.left, int(bird_y)), 24, 3)
        pygame.draw.circle(screen, WHITE, (bird_x + game_rect.left + 10, int(bird_y) - 6), 8)
        pygame.draw.circle(screen, BLACK, (bird_x + game_rect.left + 12, int(bird_y) - 6), 4)
        pygame.draw.polygon(screen, ORANGE, [
            (bird_x + game_rect.left + 18, int(bird_y)),
            (bird_x + game_rect.left + 38, int(bird_y) + 8),
            (bird_x + game_rect.left + 18, int(bird_y) + 12)
        ])
        wing_y = int(bird_y) - 10 if bird_vel < 0 else int(bird_y) + 6
        pygame.draw.ellipse(screen, (255, 180, 0), (bird_x + game_rect.left - 20, wing_y, 22, 14))

    # UI texts
    if game_state == WAITING:
        text1 = title_font.render("five finger", True, (0, 100, 255))
        text2 = small_font.render("to start", True, BLACK)
        screen.blit(text1, (game_rect.centerx - text1.get_width()//2, HEIGHT//2 - 80))
        screen.blit(text2, (game_rect.centerx - text2.get_width()//2, HEIGHT//2 + 20))

    elif game_state == COUNTDOWN:
        count_text = title_font.render(str(countdown_value), True, (255, 140, 0))
        screen.blit(count_text, (game_rect.centerx - count_text.get_width()//2, HEIGHT//2 - 80))

    elif game_state == PLAYING:
        score_text = score_font.render(f"score {score}", True, BLACK)
        high_text = small_font.render(f"champion: {high_score}", True, BLACK)
        screen.blit(score_text, (game_rect.centerx - score_text.get_width()//2, 40))
        screen.blit(high_text, (game_rect.centerx - high_text.get_width()//2, 110))

    elif game_state == GAMEOVER:
        overlay = pygame.Surface((GAME_WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        screen.blit(overlay, (game_rect.left, 0))

        over_text = title_font.render("Over!", True, RED)
        final_score = score_font.render(f"Score: {score}", True, WHITE)
        wait_text = small_font.render("return", True, WHITE)

        screen.blit(over_text, (game_rect.centerx - over_text.get_width()//2, HEIGHT//2 - 80))
        screen.blit(final_score, (game_rect.centerx - final_score.get_width()//2, HEIGHT//2 - 10))
        screen.blit(wait_text, (game_rect.centerx - wait_text.get_width()//2, HEIGHT//2 + 50))

    # Webcam region
    cam_x = GAME_WIDTH + 40
    frame_resized = cv2.resize(frame, (CAM_WIDTH, HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(frame_surf, (cam_x, 0))

    pygame.draw.line(screen, BLACK, (cam_x - 10, 0), (cam_x - 10, HEIGHT), 5)

    # Labels
    game_label = title_font.render("Game", True, BLACK)
    cam_label_bg = pygame.Surface((220, 60), pygame.SRCALPHA)
    cam_label_bg.fill((0, 0, 0, 160))
    cam_label = title_font.render("CAMERA", True, WHITE)

    screen.blit(game_label, (game_rect.centerx - game_label.get_width()//2, HEIGHT - 90))
    screen.blit(cam_label_bg, (cam_x + CAM_WIDTH//2 - 110, HEIGHT - 90))
    screen.blit(cam_label, (cam_x + CAM_WIDTH//2 - cam_label.get_width()//2, HEIGHT - 80))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
cap.release()
pose.close()
hands.close()
pygame.quit()