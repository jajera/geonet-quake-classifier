#!/usr/bin/env python3
"""Day 3 silent demo — before/after composites + Kiro conversation focus."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"
W, H = 1440, 1080
BG = (12, 14, 18)
CARD_BG = (18, 20, 26)
WHITE = (245, 248, 250)
MUTED = (160, 168, 176)
ACCENT = (56, 189, 168)
RED = (220, 80, 80)
GREEN = (70, 170, 110)
LINE = (40, 44, 52)
OUT = HERE / "day-03-fix-demo.mp4"


def font(size: int, bold: bool = False):
    candidates = [
        (
            "/usr/share/fonts/montserrat-fonts/Montserrat-Bold.ttf"
            if bold
            else "/usr/share/fonts/montserrat-fonts/Montserrat-Regular.ttf"
        ),
        (
            "/usr/share/fonts/google-noto/NotoSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/google-noto/NotoSans-Regular.ttf"
        ),
        (
            "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"
        ),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered(draw, lines, gap: int = 16, y_offset: int = 0) -> None:
    measured = []
    total_h = 0
    for text, fnt, color in lines:
        bbox = draw.textbbox((0, 0), text, font=fnt)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        measured.append((text, fnt, color, tw, th))
        total_h += th + gap
    total_h -= gap
    y = (H - total_h) // 2 + y_offset
    for text, fnt, color, tw, th in measured:
        draw.text(((W - tw) // 2, y), text, font=fnt, fill=color)
        y += th + gap


def black_card(path: Path, lines, top_color=ACCENT) -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 8], fill=top_color)
    draw.rectangle([0, H - 8, W, H], fill=top_color)
    draw_centered(draw, lines)
    img.save(path)


def fit(src: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img = src.convert("RGBA")
    scale = min(max_w / img.width, max_h / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def rounded_panel(img: Image.Image, radius: int = 12) -> Image.Image:
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0))
    out.putalpha(mask)
    return out


def frame_border(img: Image.Image, color=(60, 64, 72), width: int = 2) -> Image.Image:
    bordered = ImageOps.expand(img.convert("RGB"), border=width, fill=color)
    return bordered.convert("RGBA")


def make_before_composite(path: Path) -> None:
    """Dashboard left + two 404 callouts right (improved annotated before)."""
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_f = font(36, True)
    body_f = font(20)
    small_f = font(16)

    draw.text((48, 36), "BEFORE — Earthquake Maps", font=title_f, fill=(30, 30, 30))
    draw.text(
        (48, 84),
        "Looks fine… until you click some model links",
        font=body_f,
        fill=(90, 90, 90),
    )

    dash = fit(Image.open(HERE / "before-01-dashboard.png"), 720, 820)
    dash = rounded_panel(dash, 10)
    canvas.paste(dash, (48, 130), dash)

    err1 = fit(Image.open(HERE / "before-02-404-decision-tree.png"), 520, 280)
    err1 = frame_border(err1, (200, 80, 80), 3)
    err1 = rounded_panel(err1, 8)
    canvas.paste(err1, (820, 160), err1)
    draw.text((820, 130), "Click: Decision Tree Model", font=small_f, fill=RED)

    err2 = fit(Image.open(HERE / "before-03-404-transformer.png"), 520, 280)
    err2 = frame_border(err2, (200, 80, 80), 3)
    err2 = rounded_panel(err2, 8)
    canvas.paste(err2, (820, 520), err2)
    draw.text((820, 490), "Click: Transformer Model Visualization", font=small_f, fill=RED)

    # connector lines (simple)
    draw.line([(760, 300), (820, 300)], fill=(120, 120, 120), width=2)
    draw.line([(760, 660), (820, 660)], fill=(120, 120, 120), width=2)

    canvas.save(path)


def make_after_composite(path: Path) -> None:
    """Same layout as before, but OK badges + small working sub-screenshots."""
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_f = font(36, True)
    body_f = font(20)
    small_f = font(16)

    draw.text((48, 36), "AFTER — Earthquake Maps", font=title_f, fill=(30, 30, 30))
    draw.text(
        (48, 84),
        "Same dashboard shape — links work, badges show OK + timestamps",
        font=body_f,
        fill=(90, 90, 90),
    )

    dash = fit(Image.open(HERE / "after-01-badges.png"), 720, 820)
    dash = rounded_panel(dash, 10)
    canvas.paste(dash, (48, 130), dash)

    # small working insets
    y0 = 150
    for label, fname, color in [
        ("Decision Tree Map", "thumb-decision-tree-map.png", GREEN),
        ("Decision Tree Model PNG", "thumb-decision-tree-viz.png", GREEN),
        ("Transformer Model PNG", "thumb-transformer-viz.png", GREEN),
    ]:
        thumb = fit(Image.open(HERE / fname), 520, 200)
        thumb = frame_border(thumb, color, 3)
        thumb = rounded_panel(thumb, 8)
        canvas.paste(thumb, (820, y0 + 24), thumb)
        draw.text((820, y0), f"OK · {label}", font=small_f, fill=GREEN)
        y0 += 260

    canvas.save(path)


def make_prompt_slide(path: Path) -> None:
    """Show we only gave Kiro a bit of detail."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 8], fill=ACCENT)
    draw.rectangle([0, H - 8, W, H], fill=ACCENT)

    title_f = font(42, True)
    body_f = font(26)
    mono_f = font(22)

    draw.text((80, 80), "What I asked Kiro", font=title_f, fill=WHITE)
    draw.text(
        (80, 150),
        "I only knew a little — just the symptom, not the root cause.",
        font=body_f,
        fill=MUTED,
    )

    # prompt box
    box = [80, 230, W - 80, 780]
    draw.rounded_rectangle(box, radius=16, fill=CARD_BG, outline=LINE, width=2)
    prompt = [
        "The Earthquake Maps section on index.html has",
        "broken links — some maps/models don't open.",
        "",
        "Investigate why, find the root cause, and fix it",
        "properly so the dashboard, generators, and CI",
        "stay consistent.",
        "",
        "Also show when each map was last generated",
        "and a clear status badge if something failed.",
    ]
    y = 270
    for line in prompt:
        draw.text((120, y), line, font=mono_f, fill=ACCENT if line else MUTED)
        y += 42

    draw.text(
        (80, 840),
        "Short prompt. Kiro had to investigate the rest.",
        font=body_f,
        fill=MUTED,
    )
    img.save(path)


def make_summary_slide(path: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 8], fill=ACCENT)
    draw.rectangle([0, H - 8, W, H], fill=ACCENT)
    title_f = font(48, True)
    body_f = font(28)
    draw.text((80, 120), "Summary", font=title_f, fill=WHITE)
    bullets = [
        "Old learning-project dashboard looked fine — some links 404'd.",
        "Kiro Autopilot traced dashboard vs script vs CI filenames.",
        "Fixed naming, added transformer viz, OK/Missing badges.",
        "CI regenerated the files — all 12 links green.",
    ]
    y = 240
    for b in bullets:
        draw.text((100, y), "•  " + b, font=body_f, fill=MUTED)
        y += 90
    img.save(path)


def make_lesson_slide(path: Path) -> None:
    """Vibe-coding lesson: YMMV, better questions + context = faster/cheaper."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 8], fill=ACCENT)
    draw.rectangle([0, H - 8, W, H], fill=ACCENT)
    title_f = font(44, True)
    body_f = font(26)
    small_f = font(22)

    draw.text((80, 90), "Lesson learned", font=title_f, fill=WHITE)
    draw.text(
        (80, 160),
        "Your mileage will vary with vibe coding.",
        font=body_f,
        fill=MUTED,
    )

    points = [
        (
            "Ask the right question",
            "Clear symptom + what “done” looks like.",
        ),
        (
            "Give useful context",
            "A little closer detail focuses the agent — faster,",
            "cheaper, fewer blind iterations.",
        ),
    ]
    # flatten for drawing
    y = 240
    for title, *lines in points:
        draw.text((100, y), "•  " + title, font=body_f, fill=ACCENT)
        y += 48
        for line in lines:
            draw.text((140, y), line, font=small_f, fill=MUTED)
            y += 40
        y += 28

    draw.text(
        (80, 880),
        "Better signal up front = less thrash later.",
        font=small_f,
        fill=WHITE,
    )
    img.save(path)


def make_guardrails_slide(path: Path) -> None:
    """Verify as accountability; still scrutinize the result."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 8], fill=ACCENT)
    draw.rectangle([0, H - 8, W, H], fill=ACCENT)
    title_f = font(44, True)
    body_f = font(26)
    small_f = font(22)

    draw.text((80, 90), "Guard rails", font=title_f, fill=WHITE)
    draw.text(
        (80, 160),
        "Context alone isn’t enough — make Autopilot accountable.",
        font=body_f,
        fill=MUTED,
    )

    boxes = [
        (
            "1 · Ask it to verify",
            [
                "End-to-end check · match generators,",
                "dashboard, and CI · note what’s incomplete.",
                "Turns the agent into its own peer review.",
            ],
        ),
        (
            "2 · Still scrutinize yourself",
            [
                "Trust, but open the links. Read the diff.",
                "Run it. Badges help — you remain the bar.",
            ],
        ),
    ]
    y = 250
    for title, lines in boxes:
        draw.rounded_rectangle(
            [80, y, W - 80, y + 200],
            radius=14,
            fill=CARD_BG,
            outline=LINE,
            width=2,
        )
        draw.text((110, y + 28), title, font=body_f, fill=ACCENT)
        ty = y + 78
        for line in lines:
            draw.text((110, ty), line, font=small_f, fill=MUTED)
            ty += 36
        y += 230

    img.save(path)


def make_intro_slide(path: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 8], fill=ACCENT)
    draw.rectangle([0, H - 8, W, H], fill=ACCENT)

    title_f = font(46, True)
    body_f = font(28)
    small_f = font(24)

    # optional course banner strip
    banner = HERE / "ref-genai-aws-course.png"
    if banner.exists():
        b = fit(Image.open(banner), 980, 120)
        # rotate? banner is vertical text - leave as accent strip
        canvas_b = Image.new("RGB", (W, 90), (235, 235, 235))
        # paste banner scaled into strip poorly - instead label course
        img.paste(canvas_b, (0, 40))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W, 8], fill=ACCENT)
        draw.text(
            (80, 60),
            "Introducing Generative AI with AWS  ·  learning project",
            font=small_f,
            fill=(40, 40, 40),
        )

    draw.text((80, 200), "This was my old code", font=title_f, fill=WHITE)
    draw.text(
        (80, 280),
        "from when I was learning generative AI",
        font=body_f,
        fill=MUTED,
    )
    draw.text(
        (80, 340),
        "and took this course.",
        font=body_f,
        fill=MUTED,
    )
    draw.text(
        (80, 460),
        "GeoNet Quake Classifier · Day 3",
        font=title_f,
        fill=ACCENT,
    )
    draw.text(
        (80, 540),
        "Fix a broken project with Kiro Autopilot",
        font=body_f,
        fill=WHITE,
    )
    draw.text((80, 900), "Kiro Birthday 2026", font=small_f, fill=MUTED)
    img.save(path)


def fit_on_black(src_path: Path, out_path: Path, caption: str | None = None) -> None:
    src = Image.open(src_path).convert("RGBA")
    # leave room for caption bar
    max_h = H - 90 if caption else H - 40
    fitted = fit(src, W - 80, max_h)
    canvas = Image.new("RGB", (W, H), BG)
    x = (W - fitted.width) // 2
    y = 40 if caption else (H - fitted.height) // 2
    if caption:
        y = 70
    canvas.paste(fitted, (x, y), fitted)
    if caption:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, W, 56], fill=(8, 10, 14))
        draw.text((40, 14), caption, font=font(24, True), fill=ACCENT)
    canvas.save(out_path)


def pick_encoder() -> str:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libx264" in out:
        return "libx264"
    if "libopenh264" in out:
        return "libopenh264"
    raise SystemExit("No H.264 encoder")


def build_assets() -> None:
    BUILD.mkdir(exist_ok=True)
    title_f = font(52, True)
    body_f = font(28)

    make_intro_slide(BUILD / "01-intro.png")
    black_card(
        BUILD / "02-card-before.png",
        [
            ("Before", title_f, RED),
            ("Some Earthquake Maps links returned 404", body_f, WHITE),
        ],
        top_color=RED,
    )
    make_before_composite(BUILD / "03-before-composite.png")
    make_prompt_slide(BUILD / "04-prompt.png")
    black_card(
        BUILD / "05-card-kiro.png",
        [
            ("Kiro Autopilot", title_f, ACCENT),
            ("Investigation → fix → verify → postmortem", body_f, WHITE),
        ],
    )
    fit_on_black(
        HERE / "kiro-01-investigation-and-transformer-fix.png",
        BUILD / "06-kiro-01.png",
        "Kiro · found filename mismatch + missing transformer PNG",
    )
    fit_on_black(
        HERE / "kiro-02-summary-diff-stat.png",
        BUILD / "07-kiro-02.png",
        "Kiro · applied fix across scripts, dashboard, requirements",
    )
    fit_on_black(
        HERE / "kiro-03-verify-commands.png",
        BUILD / "08-kiro-03.png",
        "Kiro · verified generators match index.html and CI",
    )
    fit_on_black(
        HERE / "kiro-04-verify-and-postmortem.png",
        BUILD / "09-kiro-04.png",
        "Kiro · verification matrix + README postmortem",
    )
    black_card(
        BUILD / "10-card-after.png",
        [
            ("After", title_f, ACCENT),
            ("Same shape of dashboard — now green", body_f, WHITE),
        ],
    )
    make_after_composite(BUILD / "11-after-composite.png")
    make_summary_slide(BUILD / "12-summary.png")
    make_lesson_slide(BUILD / "13-lesson.png")
    make_guardrails_slide(BUILD / "14-guardrails.png")
    black_card(
        BUILD / "15-end.png",
        [
            ("Built with Kiro", title_f, WHITE),
            ("#BuildWithKiro  #TeamKiro  @kirodotdev", body_f, MUTED),
        ],
    )


def run_ffmpeg(encoder: str) -> None:
    # ~80–95s — ending carries vibe-coding lesson + guard rails
    inputs = [
        ("01-intro.png", 5),
        ("02-card-before.png", 3),
        ("03-before-composite.png", 6),
        ("04-prompt.png", 6),
        ("05-card-kiro.png", 3),
        ("06-kiro-01.png", 7),
        ("07-kiro-02.png", 7),
        ("08-kiro-03.png", 6),
        ("09-kiro-04.png", 6),
        ("10-card-after.png", 3),
        ("11-after-composite.png", 7),
        ("12-summary.png", 5),
        ("13-lesson.png", 7),
        ("14-guardrails.png", 7),
        ("15-end.png", 3),
    ]

    cmd: list[str] = ["ffmpeg", "-y"]
    for name, secs in inputs:
        cmd += ["-loop", "1", "-t", str(secs), "-i", name]

    n = len(inputs)
    filters = [
        f"[{i}:v]scale={W}:{H},setsar=1,fps=30,format=yuv420p[v{i}]"
        for i in range(n)
    ]
    concat_in = "".join(f"[v{i}]" for i in range(n))
    filters.append(f"{concat_in}concat=n={n}:v=1:a=0[outv]")

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUT),
    ]
    if encoder == "libx264":
        i = cmd.index("libx264")
        cmd[i + 1 : i + 1] = ["-preset", "medium", "-crf", "20"]
    else:
        cmd += ["-b:v", "2500k"]

    total = sum(s for _, s in inputs)
    print(f"Encoding {n} segments (~{total}s) → {OUT.name}")
    subprocess.run(cmd, cwd=BUILD, check=True)
    print(f"Wrote {OUT}")


def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    encoder = pick_encoder()
    print(f"Using encoder: {encoder}")
    build_assets()
    run_ffmpeg(encoder)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
