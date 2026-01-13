import cv2
import mediapipe as mp
import pygame
import random
import time
import threading
import numpy as np

# --- CẤU HÌNH ---
WIDTH, HEIGHT = 900, 600
GAME_WIDTH = 500
CAM_WIDTH = 400
FPS = 60

# Vật lý Easy Mode
GRAVITY = 0.2
FLAP_POWER = -5
PIPE_SPEED = 2
PIPE_GAP = 250
PIPE_DISTANCE = 400

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("Arial", 40, bold=True)
font_small = pygame.font.SysFont("Arial", 25)


class VisionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

        # Khởi tạo cả Pose và Hands
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

        self.frame = None
        self.flap_event = False
        self.fingers_count = 0
        self.running = True
        self.last_flap_time = 0

    def run(self):
        prev_y = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Xử lý Pose (để nhảy)
            pose_results = self.pose.process(rgb)
            if pose_results.pose_landmarks:
                wrist = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                curr_y = wrist.y
                if (curr_y - prev_y > 0.05) and (time.time() - self.last_flap_time > 0.5):
                    self.flap_event = True
                    self.last_flap_time = time.time()
                prev_y = curr_y
                mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks,
                                                          self.mp_pose.POSE_CONNECTIONS)

            # 2. Xử lý Hands (để bắt đầu game)
            hand_results = self.hands.process(rgb)
            self.fingers_count = 0
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Đếm số ngón tay mở
                    # Các mốc đầu ngón tay: 8, 12, 16, 20 (Trỏ, Giữa, Áp út, Út)
                    finger_tips = [8, 12, 16, 20]
                    count = 0
                    for tip in finger_tips:
                        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                            count += 1
                    # Ngón cái (xử lý riêng theo chiều ngang)
                    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                        count += 1
                    self.fingers_count = count
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            self.frame = frame

    def stop(self):
        self.running = False
        self.cap.release()


vision = VisionThread()
vision.start()

# --- TRẠNG THÁI GAME ---
# Trạng thái: "START_SCREEN", "PLAYING", "GAME_OVER"
game_state = "START_SCREEN"
bird_y = HEIGHT // 2
bird_vel = 0
pipes = []
score = 0
start_counter = 0  # Để đếm thời gian giữ 5 ngón tay


def reset_game():
    global bird_y, bird_vel, pipes, score, game_state
    bird_y = HEIGHT // 2
    bird_vel = 0
    pipes = [{"x": GAME_WIDTH + 100, "top": random.randint(50, 200), "passed": False}]
    score = 0
    game_state = "PLAYING"


# --- VÒNG LẶP CHÍNH ---
running = True
while running:
    screen.fill((240, 240, 240))
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # 1. LOGIC BẮT ĐẦU GAME BẰNG 5 NGÓN TAY
    if game_state in ["START_SCREEN", "GAME_OVER"]:
        if vision.fingers_count == 5:
            start_counter += 1
        else:
            start_counter = 0

        if start_counter > 30:  # Giữ 5 ngón tay trong ~0.5 giây (30 frames)
            reset_game()
            start_counter = 0

    # 2. LOGIC TRONG KHI CHƠI
    if game_state == "PLAYING":
        if vision.flap_event:
            bird_vel = FLAP_POWER
            vision.flap_event = False

        bird_vel += GRAVITY
        bird_y += bird_vel

        for pipe in pipes:
            pipe["x"] -= PIPE_SPEED
        if pipes[-1]["x"] < GAME_WIDTH - PIPE_DISTANCE:
            pipes.append({"x": GAME_WIDTH, "top": random.randint(50, HEIGHT - PIPE_GAP - 50), "passed": False})
        if pipes[0]["x"] < -60:
            pipes.pop(0)

        # Va chạm
        bird_rect = pygame.Rect(100, bird_y, 30, 30)
        for pipe in pipes:
            if bird_rect.colliderect(pygame.Rect(pipe["x"], 0, 60, pipe["top"])) or \
                    bird_rect.colliderect(pygame.Rect(pipe["x"], pipe["top"] + PIPE_GAP, 60, HEIGHT)):
                game_state = "GAME_OVER"
            if not pipe["passed"] and pipe["x"] < 100:
                score += 1
                pipe["passed"] = True
        if bird_y > HEIGHT or bird_y < 0:
            game_state = "GAME_OVER"

    # 3. VẼ GIAO DIỆN
    # Vẽ nền và ống
    pygame.draw.rect(screen, (200, 230, 255), (0, 0, GAME_WIDTH, HEIGHT))
    for pipe in pipes:
        pygame.draw.rect(screen, (46, 204, 113), (pipe["x"], 0, 60, pipe["top"]))
        pygame.draw.rect(screen, (46, 204, 113), (pipe["x"], pipe["top"] + PIPE_GAP, 60, HEIGHT))
    pygame.draw.circle(screen, (231, 76, 60), (100, int(bird_y)), 15)

    # Hiển thị thông báo theo trạng thái
    if game_state == "START_SCREEN":
        txt = font_big.render("GIƠ 5 NGÓN TAY", True, (0, 0, 255))
        screen.blit(txt, (70, HEIGHT // 2 - 50))
        sub_txt = font_small.render("Để bắt đầu trò chơi", True, (0, 0, 0))
        screen.blit(sub_txt, (140, HEIGHT // 2 + 20))

    elif game_state == "GAME_OVER":
        txt = font_big.render("GAME OVER", True, (255, 0, 0))
        screen.blit(txt, (130, HEIGHT // 2 - 50))
        sub_txt = font_small.render("Giơ 5 ngón tay để thử lại", True, (0, 0, 0))
        screen.blit(sub_txt, (100, HEIGHT // 2 + 20))

    # Vẽ Camera và thông tin ngón tay
    if vision.frame is not None:
        cam_f = cv2.resize(vision.frame, (CAM_WIDTH, HEIGHT))
        cam_f = cv2.cvtColor(cam_f, cv2.COLOR_BGR2RGB)
        cam_f = np.rot90(cam_f)
        cam_surface = pygame.surfarray.make_surface(cv2.flip(cam_f, 0))
        screen.blit(cam_surface, (GAME_WIDTH, 0))

        # Hiển thị số ngón tay đang đếm được lên màn hình camera
        f_txt = font_big.render(f"Fingers: {vision.fingers_count}", True, (255, 255, 0))
        screen.blit(f_txt, (GAME_WIDTH + 20, 20))

        # Vẽ thanh tiến trình "Loading" khi giơ 5 ngón tay
        if start_counter > 0:
            pygame.draw.rect(screen, (0, 255, 0), (GAME_WIDTH, HEIGHT - 10, start_counter * (CAM_WIDTH / 30), 10))

    pygame.display.flip()
    clock.tick(FPS)

vision.stop()
pygame.quit()