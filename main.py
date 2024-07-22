import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import argparse
import ast
from matplotlib.colors import hsv_to_rgb

pygame.mixer.init(frequency=44100, size=-16, channels=1, allowedchanges=0)

# Define constants
RADIUS_BIG_CIRCLE = 10
RADIUS_BALL = 0.5
dt = 0.01
GRAVITY = -9.81

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Ball animation with sound.')
parser.add_argument('--positions', type=str, default='[-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]', help='Array of x positions for the balls')
args = parser.parse_args()

# Convert the input string to a list of floats
x_positions = ast.literal_eval(args.positions)

# Initial conditions
num_balls = len(x_positions)
ball_positions = np.array([[x, 0.0] for x in x_positions])
ball_velocities = np.array([[0.0, 1.5] for _ in range(num_balls)])
ball_colors = ['blue'] * num_balls

# Define pentatonic scale frequencies across two octaves (C3 to C5 pentatonic)
base_frequencies = np.array([130.81, 146.83, 164.81, 196.00, 220.00, 261.63])  # C3 to C4 pentatonic
pentatonic_frequencies = np.concatenate([
    np.linspace(base_frequencies[i], base_frequencies[i + 1], 3)[:-1] for i in range(len(base_frequencies) - 1)
] + [np.array([base_frequencies[-1]])])
pentatonic_frequencies = np.concatenate([pentatonic_frequencies * (2 ** i) for i in range(2)])  # C3 to C5

# Generate colors from red to purple
colors = hsv_to_rgb(np.column_stack((np.linspace(0, 0.833, len(pentatonic_frequencies)), np.ones(len(pentatonic_frequencies)), np.ones(len(pentatonic_frequencies)))))

# Precompute sine waves and convert to pygame Sound objects
sounds = {}
sample_rate = 44100
duration = 0.7
fade_in_time = 0.02
fade_out_time = 0.02
fade_in_samples = int(sample_rate * fade_in_time)
fade_out_samples = int(sample_rate * fade_out_time)
total_samples = int(sample_rate * duration)

for freq in pentatonic_frequencies:
    t = np.linspace(0, duration, total_samples, False)
    wave = (0.1 * np.sin(freq * 2 * np.pi * t) * 32767).astype(np.int16)

    # Apply fade-in and fade-out
    fade_in_envelope = np.linspace(0, 1, fade_in_samples)
    fade_out_envelope = np.linspace(1, 0, fade_out_samples)
    envelope = np.ones(total_samples)
    envelope[:fade_in_samples] = fade_in_envelope
    envelope[-fade_out_samples:] = fade_out_envelope
    wave = (wave * envelope).astype(np.int16)

    wave = wave.flatten()
    sound = pygame.sndarray.make_sound(wave)
    sounds[freq] = sound

# Function to map x-coordinate to nearest pentatonic frequency and corresponding color
def map_x_to_frequency_and_color(x):
    proportion = (x + RADIUS_BIG_CIRCLE) / (2 * RADIUS_BIG_CIRCLE)
    index = int(proportion * (len(pentatonic_frequencies) - 1))
    return pentatonic_frequencies[index], colors[index]

# Function to play note based on collision position and change ball color
def play_note_from_collision(ball_index, ball_position):
    x, y = ball_position
    if y <= 0:
        freq, color = map_x_to_frequency_and_color(x)
        sounds[freq].play()
        ball_colors[ball_index] = color

# Function to check collision with the big circle and reflect velocity
def check_collision_and_reflect(ball_index, ball_position, ball_velocity):
    distance_from_center = np.linalg.norm(ball_position)
    if distance_from_center + RADIUS_BALL >= RADIUS_BIG_CIRCLE:
        normal = ball_position / distance_from_center
        ball_velocity = ball_velocity - 2 * np.dot(ball_velocity, normal) * normal
        play_note_from_collision(ball_index, ball_position)
    return ball_velocity

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-RADIUS_BIG_CIRCLE, RADIUS_BIG_CIRCLE)
ax.set_ylim(-RADIUS_BIG_CIRCLE, RADIUS_BIG_CIRCLE)
big_circle = plt.Circle((0, 0), RADIUS_BIG_CIRCLE, fill=False)
ax.add_artist(big_circle)
balls = [plt.Circle(ball_positions[i], RADIUS_BALL) for i in range(num_balls)]
for ball in balls:
    ax.add_artist(ball)

# Update function for animation
def update(frame):
    global ball_positions, ball_velocities, ball_colors
    for i in range(num_balls):
        # Update velocity with gravity
        ball_velocities[i][1] += GRAVITY * dt
        # Update position
        ball_positions[i] += ball_velocities[i] * dt
        # Check for collision and reflect velocity if needed
        ball_velocities[i] = check_collision_and_reflect(i, ball_positions[i], ball_velocities[i])
        balls[i].center = ball_positions[i]
        balls[i].set_color(ball_colors[i])
    return balls

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=1000, interval=dt*100, blit=True)

# Display the animation
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Quit pygame mixer
pygame.mixer.quit()
