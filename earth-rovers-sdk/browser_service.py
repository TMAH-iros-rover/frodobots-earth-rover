import os
import time

from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "png")
QUALITY = float(os.getenv("IMAGE_QUALITY", "1.0"))
HAS_REAR_CAMERA = os.getenv("HAS_REAR_CAMERA", "False").lower() == "true"

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")


class BrowserService:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.default_viewport = {"width": 3840, "height": 2160}

    async def initialize_browser(self):
        if self.browser:
            return

        try:
            executable_path = os.getenv("CHROME_EXECUTABLE_PATH", "/usr/bin/chromium")
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                executable_path=executable_path,
                headless=True,
                args=[
                    "--ignore-certificate-errors",
                    "--no-sandbox",
                    "--autoplay-policy=no-user-gesture-required",
                    "--use-fake-ui-for-media-stream",
                    "--disable-application-cache",
                    "--disk-cache-size=0",
                    f"--window-size={self.default_viewport['width']},{self.default_viewport['height']}",
                ],
            )
            self.context = await self.browser.new_context(
                viewport=self.default_viewport,
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            self.page = await self.context.new_page()
            await self.page.goto("http://127.0.0.1:8000/sdk", wait_until="networkidle")
            await self.page.click("#join")
            await self.page.wait_for_selector("video")
            await self.page.wait_for_selector("#map")
            await self.page.wait_for_timeout(2000)
            await self.page.evaluate(
                """({ imageFormat, imageQuality }) => {
                    window.initializeImageParams({ imageFormat, imageQuality });
                }""",
                {"imageFormat": FORMAT, "imageQuality": QUALITY},
            )
        except Exception as e:
            print(f"Error initializing browser: {e}")
            await self.close_browser()
            raise

    async def take_screenshot(self, video_output_folder: str, elements: list):
        await self.initialize_browser()

        dimensions = await self.page.evaluate(
            """() => {
            return {
                width: Math.max(document.documentElement.scrollWidth, window.innerWidth),
                height: Math.max(document.documentElement.scrollHeight, window.innerHeight),
            }
        }"""
        )

        if (
            dimensions["width"] > self.default_viewport["width"]
            or dimensions["height"] > self.default_viewport["height"]
        ):
            await self.page.set_viewport_size(dimensions)

        element_map = {"front": "#player-1000", "rear": "#player-1001", "map": "#map"}

        screenshots = {}
        for name in elements:
            if name in element_map:
                element_id = element_map[name]
                output_path = f"{video_output_folder}/{name}.png"
                element = self.page.locator(element_id)
                if await element.count():
                    start_time = time.time()  # Start time
                    await element.screenshot(path=output_path)
                    end_time = time.time()  # End time
                    elapsed_time = (
                        end_time - start_time
                    ) * 1000  # Convert to milliseconds
                    print(f"Screenshot for {name} took {elapsed_time:.2f} ms")
                    screenshots[name] = output_path
                else:
                    print(f"Element {element_id} not found")
            else:
                print(f"Invalid element name: {name}")

        return screenshots

    async def data(self) -> dict:
        await self.initialize_browser()

        bot_data = await self.page.evaluate(
            """() => {
        return window.rtm_data;
        }"""
        )

        return bot_data

    async def front(self) -> str:
        await self.initialize_browser()

        front_frame = await self.page.evaluate(
            """() => {
        return getLastBase64Frame(1000) || null;
        }"""
        )

        return front_frame

    async def rear(self) -> str:
        await self.initialize_browser()

        rear_frame = await self.page.evaluate(
            """() => {
        return getLastBase64Frame(1001) || null;
        }"""
        )

        return rear_frame

    async def send_message(self, message: dict):
        await self.initialize_browser()

        await self.page.evaluate(
            """(message) => {
                window.sendMessage(message);
            }""",
            message,
        )

    async def speak(self, audio_url: str):
        await self.initialize_browser()

        result = await self.page.evaluate(
            """async (audioUrl) => {
                return await window.playAudioToRover(audioUrl);
            }""",
            audio_url,
        )

        return result

    async def close_browser(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
