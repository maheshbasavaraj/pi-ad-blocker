# Dataset Analysis Report

## Available Public Datasets for Advertisement Detection

### 1. Open Images Dataset v7

| Metric | Value |
|--------|-------|
| Total images | 9 million |
| Annotated classes | 600 |
| Ad-relevant classes | Billboard, poster, signboard |
| Billboards labeled | 600+ |
| Format | CSV annotations, bounding boxes |
| URL | https://storage.googleapis.com/openimages/web/index.html |

**Pros:**
- Large, diverse dataset with real-world scenes
- Billboard and sign classes present
- Used by many published papers

**Cons:**
- Limited billboard-specific annotations compared to total dataset
- No dedicated "advertisement" class — must combine multiple sign classes

### 2. Mapillary Vistas Dataset

| Metric | Value |
|--------|-------|
| Total images | 25,000 |
| Annotated classes | 124 (v2.0) / 147 (research) |
| Sign-related classes | 46+ (traffic signs, advertisement signs, billboards) |
| Region coverage | Global (6 continents) |
| Format | JSON, instance segmentation + detection |
| URL | https://www.mapillary.com/dataset/vistas |

**Pros:**
- Excellent variety of billboard/sign types
- Instance segmentation masks available
- Global geographic diversity

**Cons:**
- Requires license agreement for research use
- Primarily street-view perspective

### 3. BDD100K (Berkeley DeepDrive)

| Metric | Value |
|--------|-------|
| Total images | 100,000 |
| Annotated classes | 10 (detection subset) |
| Billboard/sign class | Yes — "traffic sign" and "billboard" in full set |
| Format | JSON |
| URL | https://www.visualqa.org/ |

**Pros:**
- Large, diverse, well-organized
- Includes driving videos (relevant for roadside ads)
- Multiple annotation types (detection, segmentation, tracking)

**Cons:**
- Billboard and sign annotations part of extended set, not primary
- Focused on autonomous driving (vehicle-centric perspective)

### 4. Roboflow Public Datasets

Search results for "billboard" and "advertisement":

| Dataset | Images | Classes | Format |
|---------|--------|---------|--------|
| Billboard Detection | ~3,000 | billboard | YOLO/COCO |
| Outdoor Advertising | ~2,000 | billboard, poster, digital ad | YOLO |
| Street Signs Extended | ~5,000 | sign, billboard, ad | COCO |

**Pros:**
- Pre-formatted for YOLO training
- Ready-to-download with API key
- Often includes test/train splits

**Cons:**
- Variable quality
- Some datasets are small
- May require Roboflow subscription

## Recommendations

### Primary Dataset Strategy

**Use Open Images v7** as primary training source:
- Filter for classes: `Billboard`, `Sign`, `Poster`
- Extract subset of 2,000-5,000 images with these classes
- Supplement with Mapillary Vistas for geographic diversity

### If Custom Labeling Required

1. **Collect** 5,000+ street-scene images from diverse environments
2. **Label classes**:
   ```
   billboard       — Large roadside structures (>3m²)
   digital_display — LED/LCD advertising screens
   poster          — Paper-based posted advertisements
   shop_sign       — Storefront promotional displays
   banner          — Temporary hanging banners
   hoarding        — Construction site advertisement panels
   transit_ad      — Advertisements on buses/taxis/transit
   ```

3. **Use Label Studio** or CVAT for annotation
4. **Export** in YOLO format

### Annotation Schema

```yaml
annotation_schema:
  format: "yolo"
  image_size: "variable (640x480 recommended)"
  classes:
    0: billboard
    1: digital_display
    2: poster
    3: shop_sign
    4: banner
    5: hoarding
    6: transit_ad
  annotations_per_image: "average 2-5"
  min_bbox_area: "0.5% of image"
```

## Estimated Timeline (Custom Dataset)

| Phase | Effort | Duration |
|-------|--------|----------|
| Image collection | 2 people | 1-2 weeks |
| Data cleaning | 1 person | 3-5 days |
| Annotation (per image) | ~30 seconds | 40-80 hours |
| Quality review | 1 person | 1 week |
| Split + format | 1 person | 1 day |
| **Total** | | **3-4 weeks** |

## Dataset Splits

```
Dataset Size: 5,000 images
├── Train:    3,500 (70%)
├── Val:      1,000 (20%)
└── Test:       500 (10%)
```