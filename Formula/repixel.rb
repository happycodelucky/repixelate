class Repixel < Formula
  desc "Recover true-resolution pixel art from upscaled or AI-generated images and video"
  homepage "https://github.com/happycodelucky/repixel"
  url "https://github.com/happycodelucky/repixel/archive/refs/tags/v0.3.1.tar.gz"
  sha256 "f833c8fa49e81b9c9cafd2a35873504c4f5daf701c506a298fad497131b8a821"
  # PolyForm Shield License 1.0.0 (see LICENSE.md), which has no SPDX identifier.
  license :cannot_represent
  head "https://github.com/happycodelucky/repixel.git", branch: "main"

  depends_on "numpy"
  depends_on "pillow"
  depends_on "python@3.14"
  # Only video needs ffmpeg; still images work without it.
  depends_on "ffmpeg" => :recommended

  def python
    Formula["python@3.14"].opt_bin/"python3.14"
  end

  def install
    # The tool is one module that needs only numpy and Pillow, both Homebrew formulae,
    # so it runs straight from Homebrew's Python with no virtualenv to build.
    libexec.install "repixel.py"
    (bin/"repixel").write <<~SH
      #!/bin/bash
      exec "#{python}" "#{libexec}/repixel.py" "$@"
    SH
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/repixel --version")

    system python, "-c", <<~PYTHON
      from PIL import Image
      img = Image.new("RGB", (12, 12))
      for y in range(12):
          for x in range(12):
              img.putpixel((x, y), ((x * 20) % 256, (y * 20) % 256, 128 if (x + y) % 3 else 0))
      img.resize((120, 120), Image.NEAREST).save("in.png")
    PYTHON
    output = shell_output("#{bin}/repixel in.png -o out.png --no-ask")
    assert_match "pixel size 10.00", output
    assert_match "12x12", output
    assert_path_exists testpath/"out.png"
  end
end
