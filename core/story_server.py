"""
core/story_server.py - HTTP Story Endpoint for Live Trading Narrative

Extends Prometheus metrics server to serve /story endpoint.
"""
from http.server import BaseHTTPRequestHandler
from prometheus_client import MetricsHandler
from urllib.parse import urlparse, parse_qs
import os
from typing import List, Dict, Optional


class StoryMetricsHandler(MetricsHandler):
    """Extended Prometheus handler that also serves /story endpoint"""
    
    story_file_path: Optional[str] = None  # Set by start_metrics_server()
    
    def do_GET(self):
        """Route requests to /story or /metrics"""
        parsed = urlparse(self.path)
        
        if parsed.path == '/story':
            self.serve_story(parsed)
        elif parsed.path == '/':
            self.serve_index()
        else:
            # Default: serve /metrics (Prometheus)
            super().do_GET()
    
    def serve_index(self):
        """Serve a simple index page with links"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot - Endpoints</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1e1e1e; color: #d4d4d4; padding: 40px; }
        a { color: #569cd6; text-decoration: none; font-size: 18px; display: block; margin: 10px 0; }
        a:hover { text-decoration: underline; }
        h1 { color: #4ec9b0; }
    </style>
</head>
<body>
    <h1>🤖 Trading Bot - Available Endpoints</h1>
    <a href="/metrics">📊 Prometheus Metrics</a>
    <a href="/story">📖 Live Trading Story</a>
    <a href="/story?format=json">📋 Story (JSON API)</a>
</body>
</html>"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_story(self, parsed_url):
        """Serve story in HTML or JSON format"""
        # Parse query parameters
        params = parse_qs(parsed_url.query)
        format_type = params.get('format', ['html'])[0].lower()
        max_lines = int(params.get('lines', ['100'])[0])
        filter_type = params.get('filter', [None])[0]
        
        # Read story file
        if not self.story_file_path or not os.path.exists(self.story_file_path):
            self.send_error(404, "Story file not found")
            return
        
        try:
            lines = self._read_story_lines(self.story_file_path, max_lines, filter_type)
        except Exception as e:
            self.send_error(500, f"Error reading story: {e}")
            return
        
        # Serve in requested format
        if format_type == 'json':
            self._serve_json(lines)
        else:
            self._serve_html(lines)
    
    def _read_story_lines(self, filepath: str, max_lines: int, filter_type: Optional[str]) -> List[str]:
        """Read last N lines from story file with optional filtering"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
        except Exception as e:
            return [f"Error reading file: {e}"]
        
        # Filter if requested
        if filter_type:
            filter_upper = filter_type.upper()
            all_lines = [line for line in all_lines if filter_upper in line.upper()]
        
        # Return last N lines
        return all_lines[-max_lines:] if len(all_lines) > max_lines else all_lines
    
    def _serve_html(self, lines: List[str]):
        """Render story as HTML"""
        # Get symbol from filename if possible
        symbol = "UNKNOWN"
        if self.story_file_path:
            # Extract from filename like "story_ethbtc.txt"
            basename = os.path.basename(self.story_file_path)
            if basename.startswith('story_'):
                symbol = basename.replace('story_', '').replace('.txt', '').upper()
        
        # Render lines with color coding
        rendered_lines = []
        for line in lines:
            css_class = self._detect_event_class(line)
            escaped_line = line.replace('<', '&lt;').replace('>', '&gt;')
            rendered_lines.append(f'<div class="event {css_class}">{escaped_line}</div>')
        
        story_html = '\n'.join(rendered_lines)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Live Story - {symbol}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ 
            font-family: 'Courier New', Monaco, monospace; 
            background: #1e1e1e; 
            color: #d4d4d4; 
            padding: 20px;
            margin: 0;
        }}
        h1 {{ 
            color: #4ec9b0; 
            border-bottom: 2px solid #4ec9b0;
            padding-bottom: 10px;
        }}
        .controls {{
            margin: 20px 0;
            padding: 15px;
            background: #2d2d30;
            border-radius: 5px;
        }}
        .controls a {{
            color: #569cd6;
            text-decoration: none;
            margin-right: 20px;
            padding: 8px 15px;
            background: #37373d;
            border-radius: 3px;
            display: inline-block;
        }}
        .controls a:hover {{
            background: #3e3e42;
        }}
        #story {{
            background: #252526;
            padding: 20px;
            border-radius: 5px;
            max-width: 1400px;
        }}
        .event {{ 
            margin: 3px 0; 
            padding: 8px 12px; 
            border-left: 3px solid #444;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .ath {{ border-left-color: #4ec9b0; color: #4ec9b0; }}
        .buy {{ border-left-color: #6a9955; color: #6a9955; }}
        .sell {{ border-left-color: #f48771; color: #f48771; }}
        .safety {{ border-left-color: #f44747; background: rgba(244,71,71,0.1); color: #f44747; }}
        .phoenix {{ border-left-color: #ffa500; color: #ffa500; }}
        .summary {{ border-left-color: #569cd6; background: rgba(86,156,214,0.1); color: #569cd6; }}
        .regime {{ border-left-color: #c586c0; color: #c586c0; }}
        .startup {{ border-left-color: #4ec9b0; background: rgba(78,201,176,0.1); }}
        .info {{ border-left-color: #808080; color: #999; }}
    </style>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
        
        // Show countdown
        let seconds = 30;
        setInterval(() => {{
            seconds--;
            if (seconds <= 0) seconds = 30;
            document.getElementById('refresh-timer').textContent = seconds;
        }}, 1000);
    </script>
</head>
<body>
    <h1>📖 Live Trading Story: {symbol}</h1>
    <div class="controls">
        <a href="/story">🔄 Refresh</a>
        <a href="/story?lines=50">Last 50</a>
        <a href="/story?lines=200">Last 200</a>
        <a href="/story?format=json">JSON</a>
        <span style="color: #808080; margin-left: 20px;">
            Auto-refresh in <span id="refresh-timer">30</span>s
        </span>
    </div>
    <div id="story">
        {story_html if story_html else '<div class="info">No events logged yet.</div>'}
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def _serve_json(self, lines: List[str]):
        """Render story as JSON"""
        import json
        
        parsed_lines = []
        for line in lines:
            parsed = self._parse_story_line(line)
            if parsed:
                parsed_lines.append(parsed)
        
        # Get symbol from filename
        symbol = "UNKNOWN"
        if self.story_file_path:
            basename = os.path.basename(self.story_file_path)
            if basename.startswith('story_'):
                symbol = basename.replace('story_', '').replace('.txt', '').upper()
        
        response = {
            "symbol": symbol,
            "total_lines": len(parsed_lines),
            "lines": parsed_lines
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
    
    def _detect_event_class(self, line: str) -> str:
        """Detect CSS class from line content"""
        if '🚀' in line and 'ATH' in line:
            return 'ath'
        elif '🟢' in line or 'BUY' in line:
            return 'buy'
        elif '🔴' in line or 'SELL' in line:
            return 'sell'
        elif '🚨' in line or 'SAFETY' in line:
            return 'safety'
        elif '🔥' in line or 'PHOENIX' in line:
            return 'phoenix'
        elif '📊' in line or '📈' in line or '📉' in line or 'SUMMARY' in line:
            return 'summary'
        elif '🔄' in line or 'REGIME' in line:
            return 'regime'
        elif '🚀' in line and 'STARTED' in line:
            return 'startup'
        elif '=' in line:
            return 'info'
        else:
            return ''
    
    def _parse_story_line(self, line: str) -> Optional[Dict]:
        """Parse a story line into structured data"""
        line = line.strip()
        if not line or line.startswith('='):
            return None
        
        try:
            # Format: "YYYY-MM-DD HH:MM:SS | ICON EVENT | DETAILS"
            parts = line.split(' | ', 2)
            if len(parts) < 2:
                return None
            
            timestamp = parts[0].strip()
            event_part = parts[1].strip() if len(parts) > 1 else ''
            details = parts[2].strip() if len(parts) > 2 else ''
            
            # Extract icon and event
            event_split = event_part.split(' ', 1)
            icon = event_split[0] if len(event_split) > 0 else ''
            event = event_split[1].strip() if len(event_split) > 1 else ''
            
            # Detect event type
            event_type = self._detect_event_type(icon, event)
            
            return {
                "timestamp": timestamp,
                "icon": icon,
                "event": event,
                "details": details,
                "event_type": event_type
            }
        except Exception:
            return None
    
    def _detect_event_type(self, icon: str, event: str) -> str:
        """Detect event type from icon and event name"""
        mapping = {
            '🚀': 'ath' if 'ATH' in event else 'startup',
            '🟢': 'trade_buy',
            '🔴': 'trade_sell',
            '🚨': 'safety_breaker',
            '🔥': 'phoenix_activation',
            '📊': 'summary',
            '📈': 'summary',
            '📉': 'summary',
            '🔄': 'regime_switch',
            '🎉': 'annual_summary'
        }
        return mapping.get(icon, 'other')
