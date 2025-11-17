"""
SOP Scraper from website to store SOPs as PDF
Scrapes SOPs from https://wiki.amazowl.com/sop/seller-central/SOPs
Uses Selenium to handle JavaScript-rendered content
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from pathlib import Path
import time
import shutil

def create_sops_folder():
    """Create SOPs folder if it doesn't exist, clean if needed"""
    sops_folder = Path("sops")
    if sops_folder.exists():
        print("Cleaning existing sops folder...")
        shutil.rmtree(sops_folder)
    sops_folder.mkdir(exist_ok=True)
    return sops_folder

def setup_driver():
    """Setup Chrome driver with options"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Enable PDF printing
    chrome_options.add_experimental_option('prefs', {
        'printing.print_preview_sticky_settings.appState': '{"recentDestinations":[{"id":"Save as PDF","origin":"local","account":""}],"selectedDestinationId":"Save as PDF","version":2}',
        'savefile.default_directory': str(Path.cwd() / 'sops')
    })
    
    return webdriver.Chrome(options=chrome_options)

def sanitize_filename(filename):
    """Sanitize filename to remove invalid characters"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Remove extra spaces and limit length
    filename = ' '.join(filename.split())
    return filename[:150]

def get_sop_links(driver, base_url):
    """Get all SOP links from the main page"""
    try:
        driver.get(base_url)
        # Wait for content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
        time.sleep(2)  # Extra wait for JS
        
        # Find all links
        links = driver.find_elements(By.TAG_NAME, "a")
        sop_links = []
        
        for link in links:
            try:
                href = link.get_attribute('href')
                if href and '/sop/seller-central/' in href and '#' not in href:
                    if href not in sop_links and href != base_url:
                        sop_links.append(href)
            except:
                continue
        
        return sorted(set(sop_links))
    except Exception as e:
        print(f"Error fetching SOP links: {e}")
        return []

def save_page_as_pdf(driver, url, folder):
    """Navigate to URL and save as PDF"""
    try:
        driver.get(url)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)  # Wait for JS to fully render
        
        # Get title
        title = driver.title
        if not title or title == "":
            title = url.split('/')[-1]
        
        # Get page content for verification
        try:
            content_element = driver.find_element(By.CSS_SELECTOR, "div[slot='contents']")
            if not content_element.text.strip():
                print(f"    Warning: No visible content found")
                return None
        except:
            print(f"    Warning: Could not verify content")
        
        # Generate filename
        filename = sanitize_filename(title) + ".pdf"
        filepath = folder / filename
        
        # Print to PDF using Chrome's print functionality
        print_options = {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'preferCSSPageSize': True,
        }
        
        result = driver.execute_cdp_cmd("Page.printToPDF", print_options)
        
        # Save PDF
        import base64
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(result['data']))
        
        return filepath
        
    except TimeoutException:
        print(f"    Timeout waiting for page to load")
        return None
    except Exception as e:
        print(f"    Error saving PDF: {e}")
        return None

def main():
    base_url = "https://wiki.amazowl.com/sop/seller-central/SOPs"
    
    print("Starting SOP scraper with Selenium...")
    print(f"Target URL: {base_url}")
    
    # Create folder
    sops_folder = create_sops_folder()
    print(f"Created folder: {sops_folder}")
    
    # Setup driver
    print("\nInitializing Chrome driver...")
    driver = setup_driver()
    
    try:
        # Get all SOP links
        print("Fetching SOP links...")
        sop_links = get_sop_links(driver, base_url)
        
        if not sop_links:
            print("No SOP links found!")
            return
        
        print(f"Found {len(sop_links)} unique SOP page(s)\n")
        
        # Scrape each SOP
        successful = 0
        failed = []
        
        for i, url in enumerate(sop_links, 1):
            print(f"[{i}/{len(sop_links)}] {url}")
            
            filepath = save_page_as_pdf(driver, url, sops_folder)
            
            if filepath:
                print(f"  ✓ Saved: {filepath.name}")
                successful += 1
            else:
                print(f"  ✗ Failed")
                failed.append(url)
            
            # Be polite to the server
            time.sleep(1)
        
        print(f"\n{'='*60}")
        print(f"Scraping complete!")
        print(f"Successfully saved: {successful}/{len(sop_links)} SOPs")
        print(f"Files saved in: {sops_folder.absolute()}")
        
        if failed:
            print(f"\nFailed URLs ({len(failed)}):")
            for url in failed:
                print(f"  - {url}")
    
    finally:
        driver.quit()
        print("\nBrowser closed.")

if __name__ == "__main__":
    main()

