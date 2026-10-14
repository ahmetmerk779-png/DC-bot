""
Groq destekli sohbet Discord botu (SQLite kalıcı hafızalı).

Kurulum:
    pip install discord.py groq python-dotenv

.env dosyası::
    DISCORD_TOKEN=discord_bot_tokeniniz
    GROQ_API_KEY=groq_api_anahtariniz
    # İsteğe bağlı:
    AUTO_CHANNEL_IDS=123456789,987654321   # Bot bu kanallardaki HER mesaja cevap verir
    ALLOWED_CHANNEL_IDS=                   # Doluysa bot sadece bu kanallarda çalışır
    GROQ_MODEL=llama-3.3-70b-versatile
    COOLDOWN_SECONDS=4
    DB_PATH=bot.db                         # Render'da kalıcı disk takarsanız: /data/bot.db
"""

import asyncio
import os
import sqlite3
import time
from collections import defaultdict

import discord
from aiohttp import web
from discord import app_commands
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

# ---------------------------------------------------------------- Ayarlar
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
COOLDOWN = float(os.getenv("COOLDOWN_SECONDS", "4"))
DB_PATH = os.getenv("DB_PATH", "bot.db")
MAX_HISTORY = 20          # Modele gönderilen son mesaj sayısı
MAX_STORED = 200          # Veritabanında kanal başına saklanan en fazla mesaj
MAX_INPUT_CHARS = 1500    # Kullanıcı mesajı üst sınırı


def parse_ids(value: str) -> set[int]:
    return {int(x) for x in value.replace(" ", "").split(",") if x.isdigit()}


AUTO_CHANNELS = parse_ids(os.getenv("AUTO_CHANNEL_IDS", ""))
ALLOWED_CHANNELS = parse_ids(os.getenv("ALLOWED_CHANNEL_IDS", ""))

BASE_RULES = (
    "Bir Discord sunucusunda birden fazla kişiyle sohbet ediyorsun. "
    "Mesajlar 'İsim: mesaj' formatında gelir; kişileri isimleriyle ayırt et. "
    "Kısa ve doğal cevaplar ver (genelde 1-4 cümle), gereksiz uzatma. "
    "Kullanıcı hangi dilde yazarsa o dilde cevap ver, Türkçe yazana Türkçe cevap ver. "
    "Cevabına 'İsim:' diye başlama."
)

PERSONALITIES = {
    "samimi": "Samimi, sıcak ve rahat bir arkadaş gibi konuşuyorsun.",
    "esprili": "Esprili, hafif alaycı ama kırıcı olmayan bir sohbet arkadaşısın.",
    "resmi": "Nazik, profesyonel ve resmi bir asistansın.",
    "ogretmen": "Sabırlı bir öğretmensin; konuları basitçe, örneklerle anlatırsın.",
}
DEFAULT_PERSONALITY = "samimi"


# ---------------------------------------------------------------- Veritabanı
class Database:
    """Küçük ve hızlı işlemler için senkron sqlite3 yeterlidir."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at REAL    NOT NULL DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_channel
                ON messages (channel_id, id);

            CREATE TABLE IF NOT EXISTS settings (
                channel_id  INTEGER PRIMARY KEY,
                personality TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def add_message(self, channel_id: int, role: str, content: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages (channel_id, role, content) VALUES (?, ?, ?)",
            (channel_id, role, content),
        )
        # Eski mesajları temizle, kanal başına en fazla MAX_STORED kalsın
        self.conn.execute(
            """
            DELETE FROM messages
            WHERE channel_id = ?
              AND id NOT IN (
                  SELECT id FROM messages WHERE channel_id = ?
                  ORDER BY id DESC LIMIT ?
              )
            """,
            (channel_id, channel_id, MAX_STORED),
        )
        self.conn.commit()
        return cur.lastrowid

    def delete_message(self, message_id: int) -> None:
        self.conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        self.conn.commit()

    def get_history(self, channel_id: int, limit: int = MAX_HISTORY) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE channel_id = ? ORDER BY id DESC LIMIT ?",
            (channel_id, limit),
        ).fetchall()
        history = [{"role": r, "content": c} for r, c in reversed(rows)]
        while history and history[0]["role"] != "user":  # Pencere user ile başlasın
            history.pop(0)
        return history

    def count_messages(self, channel_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE channel_id = ?", (channel_id,)
        ).fetchone()
        return min(row[0], MAX_HISTORY)

    def clear_history(self, channel_id: int) -> None:
        self.conn.execute("DELETE FROM messages WHERE channel_id = ?", (channel_id,))
        self.conn.commit()

    def get_personality(self, channel_id: int) -> str:
        row = self.conn.execute(
            "SELECT personality FROM settings WHERE channel_id = ?", (channel_id,)
        ).fetchone()
        return row[0] if row and row[0] in PERSONALITIES else DEFAULT_PERSONALITY

    def set_personality(self, channel_id: int, personality: str) -> None:
        self.conn.execute(
            """
            INSERT INTO settings (channel_id, personality) VALUES (?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET personality = excluded.personality
            """,
            (channel_id, personality),
        )
        self.conn.commit()


db = Database(DB_PATH)

# ------------------------------------------------------------ Durum
groq = AsyncGroq()  # GROQ_API_KEY ortam değişkenini otomatik okur

locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
last_used: dict[int, float] = {}

intents = discord.Intents.default()
intents.message_content = True  # Developer Portal'dan da açın
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# ------------------------------------------------------------ Yardımcılar
def system_prompt(channel_id: int) -> str:
    return f"{PERSONALITIES[db.get_personality(channel_id)]} {BASE_RULES}"


def split_message(text: str, limit: int = 1900) -> list[str]:
    """2000 karakter sınırına göre, mümkünse satır sonundan böler."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    parts.append(text)
    return [p for p in parts if p] or ["..."]


async def should_respond(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    is_dm = isinstance(message.channel, discord.DMChannel)
    if ALLOWED_CHANNELS and not is_dm and message.channel.id not in ALLOWED_CHANNELS:
        return False

    if is_dm:
        return True
    if message.channel.id in AUTO_CHANNELS:
        return True
    if client.user in message.mentions:
        return True

    # Botun mesajına yanıt verilmişse
    ref = message.reference
    if ref and ref.message_id:
        resolved = ref.resolved
        if isinstance(resolved, discord.Message):
            return resolved.author.id == client.user.id
    return False


def clean_content(message: discord.Message) -> str:
    text = message.content
    for tag in (f"<@{client.user.id}>", f"<@!{client.user.id}>"):
        text = text.replace(tag, "")
    return text.strip()[:MAX_INPUT_CHARS]


async def ask_groq(channel_id: int) -> str:
    messages = [{"role": "system", "content": system_prompt(channel_id)}]
    messages += db.get_history(channel_id)
    response = await groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=700,
        temperature=0.8,
    )
    return (response.choices[0].message.content or "").strip()


# ------------------------------------------------------------ Olaylar
async def start_web_server():
    """
    Render'ın ücretsiz Web Service'i bir porta bağlanmayı bekler ve UptimeRobot gibi
    servislerin botu ayakta tutması için bir adrese ihtiyaç vardır.
    PORT ortam değişkeni yoksa (ör. Background Worker veya bilgisayarınız) çalışmaz.
    """
    port = os.getenv("PORT")
    if not port:
        return

    async def health(_request):
        return web.Response(text="Bot çalışıyor ✅")

    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(port)).start()
    print(f"Sağlık kontrolü sunucusu {port} portunda çalışıyor.")


@client.event
async def setup_hook():
    await start_web_server()
    await tree.sync()


@client.event
async def on_ready():
    await client.change_presence(
        activity=discord.Activity(type=discord.ActivityType.listening, name="seni | @etiketle")
    )
    print(f"{client.user} olarak giriş yapıldı. Model: {MODEL} | Veritabanı: {DB_PATH}")


@client.event
async def on_message(message: discord.Message):
    if not await should_respond(message):
        return

    content = clean_content(message)
    if not content:
        await message.reply("Efendim? 👀", mention_author=False)
        return

    # Kullanıcı başına bekleme süresi
    now = time.monotonic()
    if COOLDOWN - (now - last_used.get(message.author.id, 0)) > 0:
        try:
            await message.add_reaction("⏳")
        except discord.HTTPException:
            pass
        return
    last_used[message.author.id] = now

    channel_id = message.channel.id

    # Aynı kanalda mesajlar sırayla işlensin (hafıza karışmasın)
    async with locks[channel_id]:
        user_row = db.add_message(
            channel_id, "user", f"{message.author.display_name}: {content}"
        )

        async with message.channel.typing():
            try:
                reply = await ask_groq(channel_id)
            except Exception as e:
                db.delete_message(user_row)  # Başarısız mesajı hafızadan çıkar
                print("Groq hatası:", repr(e))
                text = str(e).lower()
                if "rate" in text or "429" in text:
                    await message.reply(
                        "Çok hızlı gidiyoruz, bir süre sonra tekrar dene 😅", mention_author=False
                    )
                else:
                    await message.reply(
                        "Bir şeyler ters gitti, birazdan tekrar dener misin?", mention_author=False
                    )
                return

        if not reply:
            db.delete_message(user_row)
            await message.reply("Buna ne diyeceğimi bilemedim 😶", mention_author=False)
            return

        db.add_message(channel_id, "assistant", reply)

    for i, chunk in enumerate(split_message(reply)):
        if i == 0:
            await message.reply(chunk, mention_author=False)
        else:
            await message.channel.send(chunk)


# ------------------------------------------------------------ Slash komutları
@tree.command(name="sifirla", description="Bu kanaldaki sohbet hafızasını temizler")
async def sifirla(interaction: discord.Interaction):
    db.clear_history(interaction.channel_id)
    await interaction.response.send_message("Hafızamı temizledim, baştan başlayalım! 🧹")


@tree.command(name="kisilik", description="Botun bu kanaldaki kişiliğini değiştirir")
@app_commands.describe(tip="Botun konuşma tarzı")
@app_commands.choices(
    tip=[
        app_commands.Choice(name="Samimi", value="samimi"),
        app_commands.Choice(name="Esprili", value="esprili"),
        app_commands.Choice(name="Resmi", value="resmi"),
        app_commands.Choice(name="Öğretmen", value="ogretmen"),
    ]
)
async def kisilik(interaction: discord.Interaction, tip: app_commands.Choice[str]):
    db.set_personality(interaction.channel_id, tip.value)
    db.clear_history(interaction.channel_id)
    await interaction.response.send_message(f"Kişilik **{tip.name}** olarak ayarlandı ✅")


@tree.command(name="bilgi", description="Bot hakkında bilgi verir")
async def bilgi(interaction: discord.Interaction):
    ch = interaction.channel_id
    embed = discord.Embed(title="🤖 Bot Bilgisi", color=discord.Color.blurple())
    embed.add_field(name="Model", value=MODEL, inline=True)
    embed.add_field(name="Kişilik", value=db.get_personality(ch).capitalize(), inline=True)
    embed.add_field(name="Hafıza", value=f"{db.count_messages(ch)}/{MAX_HISTORY} mesaj", inline=True)
    embed.add_field(
        name="Nasıl konuşurum?",
        value="Beni etiketle, mesajıma yanıt ver ya da DM at.",
        inline=False,
    )
    await interaction.response.send_message(embed=embed)


client.run(DISCORD_TOKEN)
