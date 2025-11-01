from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:5001")

    # Fill in the goal and submit the form
    page.fill("#goal", "Test the VISTA V-Loop GUI")
    page.click("button[type=submit]")

    # Wait for the task to complete and the chart to be visible
    page.wait_for_selector("#qa-chart")

    # Take a screenshot
    page.screenshot(path="jules-scratch/verification/verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
