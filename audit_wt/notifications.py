import aiohttp
from config import logger, DISCORD_WEBHOOK_URL, BOT_NAME

async def send_discord_alert(title: str, description: str, color: int = 0x00FF00):
    """
    Sends a formatted embed message to Discord using a webhook.
    """
    if not DISCORD_WEBHOOK_URL:
        return
        
    try:
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "footer": {"text": BOT_NAME}
        }
        
        payload = {"embeds": [embed]}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(DISCORD_WEBHOOK_URL, json=payload) as response:
                if response.status not in (200, 204):
                    logger.warning(f"Discord webhook failed with status {response.status}")
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")
