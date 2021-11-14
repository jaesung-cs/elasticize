#include <elasticize/gpu/graphics_shader.h>

#include <fstream>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/descriptor_set_layout.h>

namespace elastic
{
namespace gpu
{
class GraphicsShader::Impl
{
public:
  Impl() = delete;

  Impl(Engine engine, const Options& options)
    : engine_(engine)
  {
    auto device = engine_.device();

    // Pipeline layout
    vk::DescriptorSetLayout setLayout = *options.pDescriptorSetLayout;
    auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
      .setSetLayouts(setLayout);
    // TODO: push constant ranges

    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);

    // Render pass
    constexpr auto finalLayout = vk::ImageLayout::ePresentSrcKHR;
    constexpr auto depthFormat = vk::Format::eD24UnormS8Uint;
    constexpr auto samples = vk::SampleCountFlagBits::e4;

    std::vector<vk::AttachmentReference> attachmentReferences;
    std::vector<vk::AttachmentDescription> attachments;
    std::vector<vk::SubpassDescription> subpasses;

    // Multisampling requires resolve
    attachments.resize(3);
    attachments[0]
      .setFormat(options.imageFormat)
      .setSamples(samples)
      .setLoadOp(vk::AttachmentLoadOp::eClear)
      .setStoreOp(vk::AttachmentStoreOp::eStore)
      .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
      .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
      .setInitialLayout(vk::ImageLayout::eUndefined)
      .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

    attachments[1]
      .setFormat(depthFormat)
      .setSamples(samples)
      .setLoadOp(vk::AttachmentLoadOp::eClear)
      .setStoreOp(vk::AttachmentStoreOp::eDontCare)
      .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
      .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
      .setInitialLayout(vk::ImageLayout::eUndefined)
      .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    attachments[2]
      .setFormat(options.imageFormat)
      .setSamples(vk::SampleCountFlagBits::e1)
      .setLoadOp(vk::AttachmentLoadOp::eDontCare)
      .setStoreOp(vk::AttachmentStoreOp::eStore)
      .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
      .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
      .setInitialLayout(vk::ImageLayout::eUndefined)
      .setFinalLayout(finalLayout);

    attachmentReferences.resize(3);
    attachmentReferences[0]
      .setAttachment(0)
      .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    attachmentReferences[1]
      .setAttachment(1)
      .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    attachmentReferences[2]
      .setAttachment(2)
      .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    subpasses.resize(1);
    subpasses[0]
      .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
      .setColorAttachments(attachmentReferences[0])
      .setPDepthStencilAttachment(&attachmentReferences[1])
      .setResolveAttachments(attachmentReferences[2]);

    std::vector<vk::SubpassDependency> dependencies(1);
    dependencies[0]
      .setSrcSubpass(VK_SUBPASS_EXTERNAL)
      .setDstSubpass(0)
      .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
      .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
      .setSrcAccessMask({})
      .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite);

    const auto renderPassInfo = vk::RenderPassCreateInfo()
      .setAttachments(attachments)
      .setSubpasses(subpasses)
      .setDependencies(dependencies);
    renderPass_ = device.createRenderPass(renderPassInfo);

    // Shader stages
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    for (const auto& shader : options.shaders)
    {
      stages.push_back(vk::PipelineShaderStageCreateInfo()
        .setModule(engine.createShaderModule(shader.filepath))
        .setStage(shader.stage)
        .setPName("main"));
    }

    // Vertex
    std::vector<vk::VertexInputBindingDescription> bindings;
    for (const auto& binding : options.bindings)
    {
      bindings.push_back(vk::VertexInputBindingDescription()
        .setBinding(binding.binding)
        .setInputRate(vk::VertexInputRate::eVertex)
        .setStride(binding.stride));
    }

    std::vector<vk::VertexInputAttributeDescription> attributes;
    for (const auto& attribute : options.attributes)
    {
      attributes.push_back(vk::VertexInputAttributeDescription()
        .setBinding(attribute.binding)
        .setLocation(attribute.location)
        .setFormat(attribute.format)
        .setOffset(attribute.offset));
    }

    auto vertexInputState = vk::PipelineVertexInputStateCreateInfo()
      .setVertexBindingDescriptions(bindings)
      .setVertexAttributeDescriptions(attributes);

    // Topology
    auto inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo()
      .setPrimitiveRestartEnable(false)
      .setTopology(vk::PrimitiveTopology::eTriangleList);

    // Viewport
    vk::Viewport viewport;
    viewport
      .setX(0.f)
      .setY(0.f)
      .setWidth(256.f)
      .setHeight(256.f)
      .setMinDepth(0.f)
      .setMaxDepth(1.f);
    vk::Rect2D scissors{ { 0u, 0u }, { 256u, 256u } };
    auto viewportState = vk::PipelineViewportStateCreateInfo()
      .setViewports(viewport)
      .setScissors(scissors);

    // Rasterization
    auto rasterizationState = vk::PipelineRasterizationStateCreateInfo()
      .setDepthClampEnable(false)
      .setRasterizerDiscardEnable(false)
      .setPolygonMode(vk::PolygonMode::eFill)
      .setCullMode(vk::CullModeFlagBits::eBack)
      .setFrontFace(vk::FrontFace::eCounterClockwise)
      .setDepthBiasEnable(false)
      .setLineWidth(1.f);

    // Multisample
    auto multisampleState = vk::PipelineMultisampleStateCreateInfo()
      .setRasterizationSamples(samples);

    // Depth stencil
    auto depthStencilState = vk::PipelineDepthStencilStateCreateInfo()
      .setDepthTestEnable(true)
      .setDepthWriteEnable(true)
      .setDepthCompareOp(vk::CompareOp::eLess)
      .setDepthBoundsTestEnable(false)
      .setStencilTestEnable(false);

    // Color blend
    auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
      .setBlendEnable(true)
      .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
      .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
      .setColorBlendOp(vk::BlendOp::eAdd)
      .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
      .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
      .setColorBlendOp(vk::BlendOp::eAdd)
      .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    auto colorBlendState = vk::PipelineColorBlendStateCreateInfo()
      .setLogicOpEnable(false)
      .setAttachments(colorBlendAttachment)
      .setBlendConstants({ 0.f, 0.f, 0.f, 0.f });

    // Dynamic states
    std::vector<vk::DynamicState> dynamicStates{
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
    };
    auto dynamicState = vk::PipelineDynamicStateCreateInfo()
      .setDynamicStates(dynamicStates);

    auto pipelineInfo = vk::GraphicsPipelineCreateInfo()
      .setStages(stages)
      .setPVertexInputState(&vertexInputState)
      .setPInputAssemblyState(&inputAssemblyState)
      .setPViewportState(&viewportState)
      .setPRasterizationState(&rasterizationState)
      .setPMultisampleState(&multisampleState)
      .setPDepthStencilState(&depthStencilState)
      .setPColorBlendState(&colorBlendState)
      .setPDynamicState(&dynamicState)
      .setLayout(pipelineLayout_)
      .setRenderPass(renderPass_)
      .setSubpass(0);

    pipeline_ = device.createGraphicsPipeline(nullptr, pipelineInfo).value;

    // Destroy shader modules
    for (auto& stage : stages)
      device.destroyShaderModule(stage.module);
  }

  ~Impl()
  {
    auto device = engine_.device();

    device.destroyRenderPass(renderPass_);
    device.destroyPipelineLayout(pipelineLayout_);
    device.destroyPipeline(pipeline_);
  }

  auto renderPass() const noexcept { return renderPass_; }
  auto pipelineLayout() const noexcept { return pipelineLayout_; }
  auto pipeline() const noexcept { return pipeline_; }

private:
  Engine engine_;

  vk::RenderPass renderPass_;
  vk::PipelineLayout pipelineLayout_;
  vk::Pipeline pipeline_;
};

GraphicsShader::GraphicsShader(Engine engine, const Options& options)
  : impl_(std::make_shared<Impl>(engine, options))
{
}

GraphicsShader::~GraphicsShader() = default;

vk::RenderPass GraphicsShader::renderPass() const noexcept
{
  return impl_->renderPass();
}

vk::PipelineLayout GraphicsShader::pipelineLayout() const noexcept
{
  return impl_->pipelineLayout();
}

vk::Pipeline GraphicsShader::pipeline() const noexcept
{
  return impl_->pipeline();
}
}
}
