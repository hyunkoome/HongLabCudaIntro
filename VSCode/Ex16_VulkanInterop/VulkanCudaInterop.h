#pragma once

#define GLFW_INCLUDE_VULKAN

#include "WindowsSecurityAttributes.h"

#undef max
#undef min

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_timer.h"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#ifdef _WIN64
#include <vulkan/vulkan_win32.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

constexpr int WIDTH = 1920; // 원래 예제에서는 매크로 사용
constexpr int HEIGHT = 1080;
constexpr int MAX_FRAMES = 4;

class VulkanCudaInterop {
  public:
    struct QueueFamilyIndices {
        int graphicsFamily = -1;
        int presentFamily = -1;

        bool IsComplete() { return graphicsFamily >= 0 && presentFamily >= 0; }
    };

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    void LoadImageData(const std::string &filename);

    void Run();

  private:
    GLFWwindow *window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    uint8_t vkDeviceUUID[VK_UUID_SIZE];

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    VkSemaphore cudaUpdateVkSemaphore, vkUpdateCudaSemaphore;
    std::vector<VkFence> inFlightFences;

    size_t currentFrame = 0;

    bool framebufferResized = false;

#ifdef _WIN64
    PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
    PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
#else
    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR = NULL;
    PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR = NULL;
#endif

    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;

    std::vector<unsigned int> imageData;
    unsigned int imageWidth, imageHeight;
    unsigned int mipLevels = 1;
    size_t totalImageMemSize;

    // CUDA objects
    cudaExternalMemory_t cudaExtMemImageBuffer;
    cudaMipmappedArray_t cudaMipmappedImageArray, cudaMipmappedImageArrayTemp,
        cudaMipmappedImageArrayOrig;
    std::vector<cudaSurfaceObject_t> surfaceObjectList, surfaceObjectListTemp;
    cudaSurfaceObject_t *d_surfaceObjectList, *d_surfaceObjectListTemp;
    cudaTextureObject_t textureObjMipMapInput;

    cudaExternalSemaphore_t cudaExtCudaUpdateVkSemaphore;
    cudaExternalSemaphore_t cudaExtVkUpdateCudaSemaphore;
    cudaStream_t streamToRun;

    StopWatchInterface *timer = nullptr;

    void InitWindow();

    static void FramebufferResizeCallback(GLFWwindow *window, int width, int height);

    void InitVulkan();
    void InitCuda();
    void MainLoop();
    void CleanupSwapChain();
    void Cleanup();
    void RecreateSwapChain();
    void CreateInstance();
    void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    void SetupDebugMessenger();
    void CreateSurface();
    void PickPhysicalDevice();
    void GetKhrExtensionsFn();
    int SetCudaVkDevice();
    void CreateLogicalDevice();
    void CreateSwapChain();
    void CreateImageViews();
    void CreateRenderPass();
    void CreateDescriptorSetLayout();
    void CreateGraphicsPipeline();
    void CreateFramebuffers();
    void CreateCommandPool();
    void CreateTextureImage();
    void GenerateMipmaps(VkImage image, VkFormat imageFormat);

#ifdef _WIN64 // For windows
    HANDLE GetVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType);
    HANDLE
    GetVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType,
                         VkSemaphore &semVkCuda);
#else
    int GetVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType);

    int GetVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType,
                             VkSemaphore &semVkCuda);
#endif

    void CreateTextureImageView();
    void CreateTextureSampler();
    VkImageView CreateImageView(VkImage image, VkFormat format);
    void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image,
                     VkDeviceMemory &imageMemory);
    void CudaVkImportSemaphore();
    void CudaVkImportImageMem();
    void CudaUpdateVkImage();
    void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
                               VkImageLayout newLayout);
    void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void CreateVertexBuffer();
    void CreateIndexBuffer();
    void CreateUniformBuffers();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer &buffer, VkDeviceMemory &bufferMemory);
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer commandBuffer);
    void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void CreateCommandBuffers();
    void CreateSyncObjects();
    void CreateSyncObjectsExt();
    void UpdateUniformBuffer();
    void DrawFrame();
    void CudaVkSemaphoreSignal(cudaExternalSemaphore_t &extSemaphore);
    void CudaVkSemaphoreWait(cudaExternalSemaphore_t &extSemaphore);
    void SubmitVulkan(uint32_t imageIndex);
    void SubmitVulkanCuda(uint32_t imageIndex);
    VkShaderModule CreateShaderModule(const std::vector<char> &code);
    VkSurfaceFormatKHR
    ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);
    VkPresentModeKHR
    ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);
    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device);
    bool IsDeviceSuitable(VkPhysicalDevice device);
    bool CheckDeviceExtensionSupport(VkPhysicalDevice device);
    // QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);

    VulkanCudaInterop::QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
        VulkanCudaInterop::QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto &queueFamily : queueFamilies) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, this->surface, &presentSupport);

            if (queueFamily.queueCount > 0 && presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.IsComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char *> GetRequiredExtensions();
    bool CheckValidationLayerSupport();

    std::vector<char> ReadFile(const std::string &filename);
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);
};