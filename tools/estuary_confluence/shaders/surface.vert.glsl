#version 430 core
in vec2 in_position;
out vec2 screen_position;
void main() {
    screen_position = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
