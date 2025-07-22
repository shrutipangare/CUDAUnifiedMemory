#include <iostream>
#include <chrono>
#include <cmath>
// Function to add arrays
void sumVectors(float *vector1, float *vector2, int size) {
for (int idx = 0; idx < size; ++idx) {
vector2[idx] = vector1[idx] + vector2[idx];
 }
}
int main(int argc, char** argv) {
int scale = 1;
int size = 1 << 20;
if (argc == 2) {
sscanf(argv[1], "%d", &scale);
 }
 size = scale * size;
float *firstArray = (float*)malloc(size * sizeof(float));
float *secondArray = (float*)malloc(size * sizeof(float));
 // Initialize arrays
for (int idx = 0; idx < size; ++idx) {
firstArray[idx] = 1.0f;
secondArray[idx] = 2.0f;
 }
auto timeStart = std::chrono::high_resolution_clock::now();
sumVectors(firstArray, secondArray, size);
auto timeEnd = std::chrono::high_resolution_clock::now();
auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart);
std::cout << "K: " << scale << " million, " << "Time: " << (float)elapsedTime.count()/1000000 << " (sec)" << std::endl;
 // Check for errors (all values should be 3.0f)
float errorMax = 0.0f;
for (int idx = 0; idx < size; idx++) {
 errorMax = std::fmax(errorMax, std::fabs(secondArray[idx] - 3.0f));
 }
std::cout << "Max error: " << errorMax << std::endl;
 // Free memory
free(firstArray);
free(secondArray);
return 0;
}