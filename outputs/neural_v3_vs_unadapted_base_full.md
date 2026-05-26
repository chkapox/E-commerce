# Generation Run Comparison

This report uses lightweight automatic checks as a triage aid. Final claims about hallucination, OCR usefulness, and product-page quality still need manual review.

| Run | n | Tok F1 | Words | Prompt Artifacts | Repetition | Bad Strings | Title Exact | Metadata Exact | OCR Use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_v3 | 1668 | 0.243 | 29.067 | 0.000 | 0.196 | 0.000 | 0.270 | - | 0.550 |
| flan_t5_base | 1668 | 0.234 | 24.435 | 0.028 | 0.031 | 0.000 | 0.638 | - | 0.520 |

## Notes By Run

### final_v3

- path: `outputs\predictions\test_neural_marketplace_v3_full.jsonl`
- color recall: 0.336 (205 rows)
- product type recall: 0.306 (280 rows)
- exact title checks: 1668; metadata value checks: 0; OCR checks: 923

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

### flan_t5_base

- path: `outputs\predictions\test_flan_t5_base_unadapted_full.jsonl`
- color recall: 0.439 (205 rows)
- product type recall: 0.418 (280 rows)
- exact title checks: 1668; metadata value checks: 0; OCR checks: 923

Potential issue samples:

- id `B0D2ZTQVGK`
  - reference: Custom Design For Xbox Controller Battery Pack: Galaxy Pattern is perfectly customized on xbox battery pack for Controller. Unique to all rechargeable xbox battery pack in the market. Not only increases the beauty of the xbox rechargeable battery pack to make you different from others, but also brings you an unparalleled game experience. This game accessories kit is good for Xbox gamers at the party. At the same time, it is perfect for children.
  - prediction: Upgraded Rechargeable Battery Pack for Xbox Controller,2x4800mWh,Xbox One S/X/Elite - Visible image caption: a pair of black joysticks - Visual OCR text: How to Install Correctly, Corrcct, Eror, Installation, For charging; please, put, facing outward, Installed Correctly - Brand: Feirsh - Category: Games - Size or pack count: 4 x 2 x 4 inches
- id `B002MAPZYC`
  - reference: 53 GALLONS/MIN: That’s right, the 13. 0 Amp motor mulches up to 53 gallons of leaves per minute RIGHT INTO THE BAG: Disposes of leaves and debris into a bag you can attach beneath (bag not included) 11:1 MULCH RATIO: That’s 11 bags chopped down into 1, all due to the innovative blade-less flex-a-line mulching system STURDY YET PORTABLE: At 20 lbs., this tool is substantial and stays in place during use, but isn’t so heavy that you can’t move it.
  - prediction: Worx WG430 13 Amp Electric Leaf Mulcher is a 13 amp bladeless electric leaf mulcher.
- id `B000YZ7M0W`
  - reference: Constructed of durable hardwood Holds hoops, stretcher bards, and scroll frames of all sizes Adjusts for height, angle, and right or left hand use.
  - prediction: The Frank A. Edmunds Universal Craft Stand, 6111, is a universal craft stand made of wood.
- id `B0C66GFRKY`
  - reference: Modular design fits any number of wrenches in a set, Patent Pending design, Made in the USA Fits size range 6mm to 38mm or 1⁄4” to 1 1⁄2” Compact design works well with all wrench types including offset, combination, open end, stubby, and ratcheting Identification labels included to easily see wrench sizes Innovative geometry aligns wrenches for a uniform presentation, features strong magnetic mount.
  - prediction: Wrench Organizer for 40 Tools with Magnetic Base (Red) - Made in the USA - Visible image caption: a black background with text and design - Visual OCR text: MODULAR DESIGN, SNAPS TOGETHER FOR, CUSTOM CONFIGURATIONS, MADE, USA
- id `B07SRW2HG4`
  - reference: Cannon Sports track and field relay batons feature an official size of 11. 5 inches by 1. 43 inches Choose from a shiny black, purple, gold, green, silver, red, blue, or assorted color pack of 6 Our relay batons each feature anodized aluminum for a nonslip grip preventing slipping or dropping Pass our track and field batons effortlessly with a smooth design made for gentle passing Our relay baton has a clean and shiny finish making it easy to.
  - prediction: The Cannon Sports Aluminum Track Relay Baton is a lightweight aluminum track relay baton designed for use in track and field events.

