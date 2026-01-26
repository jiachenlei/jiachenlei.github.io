import os
from pathlib import Path

from playwright.sync_api import sync_playwright
import os
from pathlib import Path

def bake_static_html(md_filename):
    # 1. Read Markdown
    md_path = Path("blog") / md_filename
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        return
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    with sync_playwright() as p:
        # headless=True means no window pops up
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 2. Load Template
        template_path = Path(os.getcwd()) / "template.html"
        page.goto(template_path.as_uri(), wait_until="networkidle")

        # 3. Inject Content
        # We wrap it in a try-catch to see exactly where the JS fails
        try:
            page.evaluate("content => renderDirect(content)", md_content)
        except Exception as e:
            print(f"JS Injection Failed: {e}")
            browser.close()
            return

        # 4. Wait for zero-md to render (the 'pending' attribute is key)
        print("Rendering Markdown...")
        page.wait_for_selector("zero-md:not([pending])", timeout=10000)
        
        # 5. Extract the rendered HTML from the Shadow DOM
        # This bypasses the 'empty file' problem by grabbing the generated pixels/text
        baked_html = page.evaluate("""() => {
            const zeroMd = document.querySelector('zero-md');
            if (!zeroMd || !zeroMd.shadowRoot) return "Error: No shadow root found";
            
            const body = zeroMd.shadowRoot.querySelector('.markdown-body');
            const styles = Array.from(zeroMd.shadowRoot.querySelectorAll('style'))
                                .map(s => s.outerHTML).join('');
            
            return styles + body.outerHTML;
        }""")

        if "Error" in baked_html:
            print(baked_html)
            browser.close()
            return

        # 6. Reconstruct the final file
        # We put the 'baked' HTML into the #content div and save the whole page
        # page.evaluate(f"document.getElementById('content').innerHTML = `{baked_html}`")
        page.evaluate("""({html}) => {
            document.getElementById('content').innerHTML = html;
        }""", {"html": baked_html})
        
        # Remove the scripts so the file is truly static (no more JS needed)
        page.evaluate("document.querySelectorAll('script').forEach(s => s.remove())")

        output_name = md_filename.replace('.md', '.html')
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(page.content())

        print(f"✅ Successfully baked: {output_name}")
        browser.close()

def finalize_blog_style(md_filename):
    # This matches the output name you've been using
    html_filename = md_filename.replace('.md', '.html')
    
    if not os.path.exists(html_filename):
        print(f"Error: {html_filename} not found. Run the previous Playwright script first.")
        return

    with open(html_filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add the missing KaTeX CSS for Math and Markdown Typography
    # We insert this into the <head>
    head_extensions = """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blog - Jiachen Lei</title>
    <link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* ... your existing styles ... */

    .markdown-body {
        font-size: 1.1rem;
        line-height: 1.7;
        color: var(--text-dark);
    }
    .markdown-body h2 { margin-top: 2rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border-light); padding-bottom: 0.3rem; }
    .markdown-body p { margin-bottom: 1.5rem; }
    .markdown-body blockquote { border-left: 4px solid var(--google-blue); padding-left: 1rem; color: var(--text-gray); margin: 1.5rem 0; }
    .markdown-body code { background: #f1f3f4; padding: 2px 4px; border-radius: 4px; font-family: monospace; }
    
    .markdown-body ol, 
    .markdown-body ul {
        padding-left: 2rem;
        margin-bottom: 1.5rem;
    }

    /* 2. Add spacing between list items for better readability */
    .markdown-body li {
        margin-bottom: 0.5rem;
    }

    /* 3. Ensure nested lists are also indented */
    .markdown-body li > ol,
    .markdown-body li > ul {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
}

    /* FIX FOR SUMMARY LINE BREAK */
    details summary { 
        cursor: pointer; 
        color: var(--google-blue); 
        margin: 2rem 0;
        outline: none;
    }
    details summary h4 { 
        display: inline; /* This is the critical line */
        margin-left: 8px; /* Adds space after the arrow */
        vertical-align: middle;
    }

    .katex-display { overflow-x: auto; padding: 1rem 0; }
        
    /* Hide the default triangle */
    details summary::-webkit-details-marker { display: none; } /* Safari */
    details summary { list-style: none; } /* Chrome/Firefox */

    /* Add your own custom arrow before the text */
    details summary::before {
        content: "▶";
        font-size: 0.8rem;
        margin-right: 10px;
        transition: transform 0.2s ease;
    }

    /* Rotate the arrow when the details are open */
    details[open] summary::before {
        transform: rotate(90deg);
    }
    </style>
    """

    # 2. Inject the styles before the closing </head> tag
    final_html = content.replace('</head>', f'{head_extensions}</head>')

    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"✅ Finalized {html_filename} with Math and Typography styles.")

if __name__ == "__main__":
    bake_static_html("rfvsdm.md")
    finalize_blog_style("rfvsdm.md")