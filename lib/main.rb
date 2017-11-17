require 'opencl_ruby_ffi'
require 'narray_ffi'
require 'chunky_png'
require 'benchmark'

def render_on_gpu(width, height)
  platform = OpenCL::platforms.first
  device = platform.devices[-1]
  context = OpenCL::create_context(device)
  queue = context.create_command_queue(device, properties: OpenCL::CommandQueue::PROFILING_ENABLE)
  prog = context.create_program_with_source(File.read('lib/main.cl'))
  prog = prog.build

  image_format = OpenCL::ImageFormat::new(OpenCL::ChannelOrder::RGBA, OpenCL::ChannelType::FLOAT)
  image = context.create_image_2d(image_format, width, height)
  image_ptr = NArray.sfloat(width * height * 4)
  size = OpenCL::UInt2::new(width, height)
  seeds = NArray.sfloat(width * height).random(1)
  seeds_buf = context.create_buffer(seeds.size * seeds.element_size, flags: OpenCL::Mem::COPY_HOST_PTR, host_ptr: seeds)

  event = prog.trace(queue, [width, height], size, seeds_buf, image, local_work_size: [16, 16])
  queue.enqueue_read_image(image, image_ptr, event_wait_list: [event])
  queue.finish
  image_ptr
ensure
  puts prog.build_log
end

if __FILE__ == $0
  width, height = 512, 512

  image_ptr = nil

  puts Benchmark.measure { image_ptr = render_on_gpu(width, height) }

  png = ChunkyPNG::Image.new(width, height, ChunkyPNG::Color::TRANSPARENT)

  (0...width).each do |x|
    (0...height).each do |y|
      idx = (y * width + x) * 4
      r = image_ptr[idx]
      g = image_ptr[idx + 1]
      b = image_ptr[idx + 2]
      png[x, height - y - 1] = ChunkyPNG::Color((r * 255).round, (g * 255).round, (b * 255).round)
    end
  end

  png.save('output.png', :interlace => true)
end
