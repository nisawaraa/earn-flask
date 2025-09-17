from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd

# รายชื่อตำบล
tambon_list = [
    "ป่าแฝก", "ขุนควร", "ฝายกวาง", "ร่มเย็น", "อ่างทอง", "ผาช้างน้อย", "หนองหล่ม", "เชียงม่วน", "แม่ลาว",
    "งิม", "บ้านถ้ำ", "สบบง", "ท่าวังทอง", "ควร", "แม่กา", "ภูซาง", "จำป่าหวาย", "ทุ่งรวงทอง", "น้ำแวน",
    "บ้านต๋อม", "สระ", "คือเวียง", "หย่วน", "เจริญราษฎร์", "นาปรัง", "บ้านมาง", "ห้วยลาน", "บ้านสาง",
    "ทุ่งกล้วย", "ดอกคำใต้", "เชียงแรง", "บ้านเหล่า", "ศรีถ้อย", "ท่าจำปี", "เวียง", "บ้านต๊ำ", "ดงสุวรรณ",
    "ดงเจน", "ปง", "แม่นาเรือ", "แม่ปืม", "ดอนศรีชุม", "บุญเกิด", "บ้านปิน", "บ้านตุ่น", "สว่างอารมณ์",
    "สันโค้ง", "แม่ต๋ำ", "ทุ่งผาสุข", "เจดีย์คำ", "ป่าซาง", "สันป่าม่วง", "ออย", "เชียงบาน", "แม่สุก", "แม่อิง",
    "แม่ใจ", "ป่าสัก", "แม่ใส", "พระธาตุขิงแกง", "จุน", "ห้วยยางขาม", "ห้วยข้าวก่ำ", "ลอ", "หงส์หิน"
]

# ตั้งค่า geocoder
geolocator = Nominatim(user_agent="phayao-mapper")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

results = []

for tambon in tambon_list:
    query = f"ตำบล{tambon}, จังหวัดพะเยา, ประเทศไทย"
    location = geocode(query)
    if location:
        results.append({'ตำบล': tambon, 'lat': location.latitude, 'lon': location.longitude})
    else:
        results.append({'ตำบล': tambon, 'lat': None, 'lon': None})

df = pd.DataFrame(results)
df.to_csv("phayao_tambon_coordinates.csv", index=False)
print("✅ Done: Saved to phayao_tambon_coordinates.csv")
