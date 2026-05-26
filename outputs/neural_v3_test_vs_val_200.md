# Generation Run Comparison

This report uses lightweight automatic checks as a triage aid. Final claims about hallucination, OCR usefulness, and product-page quality still need manual review.

| Run | n | Tok F1 | Words | Prompt Artifacts | Repetition | Bad Strings | Title Exact | Metadata Exact | OCR Use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neural_v3_test | 200 | 0.250 | 29.600 | 0.000 | 0.235 | 0.000 | 0.270 | - | 0.577 |
| neural_v3_val | 200 | 0.246 | 32.300 | 0.000 | 0.240 | 0.000 | 0.270 | - | 0.469 |

## Notes By Run

### neural_v3_test

- path: `outputs\predictions\test_neural_marketplace_v3_200.jsonl`
- color recall: 0.414 (37 rows)
- product type recall: 0.306 (31 rows)
- exact title checks: 200; metadata value checks: 0; OCR checks: 104

Potential issue samples:

- id `B0BFSSBZ8V`
  - reference: Heavy-duty lifting: The SuperHandy Material Lift Stacker is an industrial-strength machine designed for heavy-duty lifting. With a loading capacity of up to 330 lbs., this lift stacker is a reliable alternative to forklifts and other freight transport machinery Versatile functionality: This material lift stacker can work in two modes, as a stacker to lift objects or as a hand truck to transfer materials. Its swivel casters and solid PU wheels.
  - prediction: SuperHandy Material Lift Winch Stacker: Superhandy material lift winch stacker is designed to provide reliable and, prioritizing safety and easy of use, lifting and lowering function, a simple but effective design features, locking mechanisms prevent.
- id `B0D2ZTQVGK`
  - reference: Custom Design For Xbox Controller Battery Pack: Galaxy Pattern is perfectly customized on xbox battery pack for Controller. Unique to all rechargeable xbox battery pack in the market. Not only increases the beauty of the xbox rechargeable battery pack to make you different from others, but also brings you an unparalleled game experience. This game accessories kit is good for Xbox gamers at the party. At the same time, it is perfect for children.
  - prediction: Upgraded Rechargeable Battery Pack for Xbox Controller,2x4800mWh Battery Pack W/RGB Charger Station,Rechargeable battery pack for Xbox Series X|S,Xbox One S/X/Elite.
- id `B07TXLZ8MR`
  - reference: ERGONOMIC DESIGN: Easy-to-squeeze, upside down design with angled neck provides postpartum relief without the mess PORTABLE BIDET: Use during postpartum, menstrual periods or as an everyday portable bidet. Holds 10oz POSTPARTUM CARE: Cleanse your perineal area during limited mobility recovery without putting pressure on soreness or c-section incision ADD TO YOUR POSTPARTUM ESSENTIALS: Start using after the first postpartum bathroom trip WHAT'S.
  - prediction: Frida Mom Upside Down Peri Bottle for Postpartum CareThe original Frida Mom upside down peri bottle for postpartum care is made of high-quality plastic and has a high-density foam. It can be used as a perineal cleansing and recovery device for new moms.
- id `B07ZSC6TNB`
  - reference: 1. Enhanced PIR Motion Inductor:Hmcity Solar Lights Outdoor bring 3 lighting modes 1. Mode 1 (Motion sensor turns the light on when people comes) 2. Mode 2-Permanent On all night(it turns on at night and stay all night.)3. Mode 3-Smart brightness control(the Light stays all night and turns brighter when it detects motion) 2. High Efficient Solar Panel:Adopted monocrystalline silicon solar cells, the conversion rate reaches up to 20. 5%.
  - prediction: Lights Reflector and 3 Lighting Modes: The HMCITY Solar Lights Outdoor 120 LED with Lights reflector and three lighting modes, Motion Sensor Wall Lights, IP65 Waterproof Solar Powered for Garden Patio Yard (2Pack) Features: High-quality, high-performance LEDs with a high-quality design and high-efficiency.
- id `B08BLYFYF6`
  - reference: Farabi Sports kids’ inner gloves are specifically designed for the fortification of hands and wrists during the combat and training sessions. Farabi Sports Neo Gel advanced glove series is a prime flex neo addition in the Boxing Protection Equipments Stitched using a Triple-needle fleece Pico-thread which strengthen the composition and make this glove extremely durable. The internal JP-Nano composition allows the air contact with the hands.
  - prediction: Hybrid Boxing GlovesFarabi Sports Kids boxing inner gloves are made of a high-quality material that provides great grip During training. They are designed to provide great grip during training.

### neural_v3_val

- path: `outputs\predictions\val_neural_marketplace_v3_200.jsonl`
- color recall: 0.205 (22 rows)
- product type recall: 0.163 (40 rows)
- exact title checks: 200; metadata value checks: 0; OCR checks: 113

Potential issue samples:

- id `B07H39S9JQ`
  - reference: High-pressure inflation mode with automatic shutoff High-volume inflation and deflation modes LED light helps to provide illumination Three power sources** for inflation on a jobsite and on the go (**20V MAX* battery, 12V DC, or 110V AC) Convenient onboard accessory storage Heavy-duty rubber feet help reduce walking when in use Threaded chuck for secure connection to inflation valve.
  - prediction: Compact and Portable: DEWALT 20V MAX Tire Inflator is a compact and portable tire inflator with an automatic shut off, LED light, and a bare tool only design. It can be used on all types of tires.
- id `B00002ND6R`
  - reference: fluidmaster400a anti-siphon toilet fill valve - fm400a This is highly durable This is manufactured in MEXICO.
  - prediction: Anti-Siphon Toilet Tank Fill Valve: The Fluidmaster 400A anti-siphon toilet tank fill valve is a universal replacement part for the Fluidmaster toilet tank Fill valve. This valve is designed to replace the existing fill valve with a new fill valve.
- id `B0064DPM64`
  - reference: AWARD-WINNING RIBS as seen on Good Morning America, Rachael Ray and Martha Stewart are fall-off-the-bone tender. Our pork baby back ribs package feeds 4 with 2 racks and a bottle of barbecue sauce SLOW-SMOKED FOR 4 HOURS after a 48-hour dry rub, Pig of the Month BBQ Ribs are succulent and flavorful. Brush-on our complimentary Love Me Tender Sauce for Memphis ribs that tantalize your taste buds PREMIUM ORGANIC MEAT is all-natural, hormone-free.
  - prediction: Slow Smoked Baby Back BBQ Ribs: Our baby back BBQ ribs are made with a slow-smoked medium-rare smoke. This is a great way to add flavor to your meal Baby Back BBQ Recipe: This baby back barbecue rib set includes 2 pieces of meat and vegetables.
- id `B07T9H5PWH`
  - reference: MADE OF TOP-LEVEL PROFESSIONAL SILICONE - In order to ensure food safety,our silicone cake molds passed the higher test European grade,LFGB approved, BPA free MAKE BAKING SAFER - High temperature resistance and anti-freezing,safe use temperature -40°C to 230°C (-40°F to 446°F), oven, microwave, freezer safe FLUTED CAKE SHAPED DESIGN - The cake tube pan lets you have more creatives with your baking,9 inches internal diameter, 3. 3 inches.
  - prediction: BPA Free Bakeware: This silicone cake mold is made of high-quality silicone, which is non-stick and non-slip. It can be used for cake, Jello, Gelatin, Bread, Monkey Bread, Brownie, Round baking pan, BPA-free bakeware, and other baked goods. It is suitable for use in cakes, jello, gelatin, bread, monkey bread, brownie, round baking pan.
- id `B0D5WLL1VL`
  - reference: Advanced technology and careful design:The Paint Sprayer,operating at 1000 W with a large capacity container,boosts painting efficiency;Its adjustable flow control suits different tasks,offering a faster,more convenient alternative to rollers and brushes Upgraded materials:The motor features a fuse with temperature protection to extend its life by cutting power when overheating;Brass nozzles outperform plastic ones,offering durability and easy.
  - prediction: High Power Electric Paint Sprayer: This high power electric paint sprayer is designed for high power and high power paint sprayers. It can be used to spray paint furniture, cabinets, fences, walls, door, garden chairs, etc. It has a high power of 1000W.

