package desu.inugram.core.sticker

import kotlin.math.*

/** Geometry uses the editor's logical pixels, independent of preview size. */
object StickerFit {
    data class Point(val x: Double, val y: Double)
    data class Rect(val left: Double, val top: Double, val right: Double, val bottom: Double)
    data class Affine(
        val a: Double = 1.0, val b: Double = 0.0, val c: Double = 0.0,
        val d: Double = 1.0, val tx: Double = 0.0, val ty: Double = 0.0,
    ) {
        fun map(x: Double, y: Double) = Point(a * x + c * y + tx, b * x + d * y + ty)
    }
    data class Layer(val pixels: IntArray, val width: Int, val height: Int, val matrix: Affine)
    data class Result(val scale: Double, val translationX: Double, val translationY: Double)

    /** Pixel runs retain holes until after the committed crop has been applied. */
    fun support(layers: List<Layer>, rectangles: List<List<Point>>, crop: Rect?, ignoreAlphaSpecks: Boolean = false): List<Point> {
        val points = ArrayList<Point>()
        fun add(polygon: List<Point>) { points.addAll(if (crop == null) polygon else clip(polygon, crop)) }
        for (layer in layers) {
            require(layer.width > 0 && layer.height > 0 && layer.pixels.size == layer.width * layer.height)
            val specks = if (ignoreAlphaSpecks) alphaSpecks(layer) else null
            fun covered(index: Int) = layer.pixels[index] ushr 24 != 0 && specks?.get(index)?.toInt() != 2
            for (y in 0 until layer.height) {
                val row = ArrayList<Point>()
                var x = 0
                while (x < layer.width) {
                    while (x < layer.width && !covered(y * layer.width + x)) x++
                    val start = x
                    while (x < layer.width && covered(y * layer.width + x)) x++
                    if (x > start) {
                        // Half a source pixel accounts for bilinear filtering at the edges.
                        val l = start - .5; val r = x + .5; val t = y - .5; val b = y + 1.5
                        val polygon = listOf(layer.matrix.map(l, t), layer.matrix.map(r, t), layer.matrix.map(r, b), layer.matrix.map(l, b))
                        row.addAll(if (crop == null) polygon else clip(polygon, crop))
                    }
                }
                // Bound memory for noisy masks with many disconnected pixel runs.
                points.addAll(hull(row))
            }
        }
        rectangles.forEach(::add)
        return hull(points)
    }

    /** Ignore only detached specks under 1% opacity, never connected edges or wholly faint layers. */
    private fun alphaSpecks(layer: Layer): ByteArray? {
        val maximumAlpha = 2
        val maximumPixels = 16
        var hasFaint = false
        var hasVisible = false
        for (pixel in layer.pixels) {
            val alpha = pixel ushr 24
            if (alpha in 1..maximumAlpha) hasFaint = true
            if (alpha > maximumAlpha) hasVisible = true
            if (hasFaint && hasVisible) break
        }
        if (!hasFaint || !hasVisible) return null
        // 0 = unvisited, 1 = retained, 2 = speck, 3 = queued in the current region.
        val states = ByteArray(layer.pixels.size)
        val queue = IntArray(maximumPixels + 1)
        for (start in layer.pixels.indices) {
            if (states[start].toInt() != 0 || layer.pixels[start] ushr 24 !in 1..maximumAlpha) continue
            queue[0] = start
            states[start] = 3
            var size = 1
            var cursor = 0
            var retain = false
            region@ while (cursor < size) {
                val index = queue[cursor++]
                val x = index % layer.width
                val y = index / layer.width
                for (ny in max(0, y - 1)..min(layer.height - 1, y + 1)) {
                    for (nx in max(0, x - 1)..min(layer.width - 1, x + 1)) {
                        val neighbor = ny * layer.width + nx
                        val alpha = layer.pixels[neighbor] ushr 24
                        if (alpha == 0) continue
                        if (alpha > maximumAlpha || states[neighbor].toInt() == 1) {
                            retain = true
                            break@region
                        }
                        if (states[neighbor].toInt() == 0) {
                            states[neighbor] = 3
                            queue[size++] = neighbor
                            if (size > maximumPixels) {
                                retain = true
                                break@region
                            }
                        }
                    }
                }
            }
            for (i in 0 until size) states[queue[i]] = if (retain) 1 else 2
        }
        return states
    }

    fun fit(points: List<Point>, angle: Double, side: Double, margin: Double = side / 512.0): Result? {
        if (points.isEmpty() || side <= 0 || !side.isFinite()) return null
        val radians = angle * PI / 180
        val c = cos(radians); val s = sin(radians)
        val rotated = points.map { Point(c * it.x - s * it.y, s * it.x + c * it.y) }
        if (rotated.any { !it.x.isFinite() || !it.y.isFinite() }) return null
        val left = rotated.minOf { it.x }; val right = rotated.maxOf { it.x }
        val top = rotated.minOf { it.y }; val bottom = rotated.maxOf { it.y }
        val cx = (left + right) / 2; val cy = (top + bottom) / 2
        val extent = max(right - left, bottom - top)
        if (extent <= 0 || margin >= side / 2) return null
        val centered = rotated.map { Point(it.x - cx, it.y - cy) }
        var lo = 0.0; var hi = side / extent
        repeat(60) {
            val mid = (lo + hi) / 2
            if (centered.all { contains(it.x * mid, it.y * mid, side, margin) }) lo = mid else hi = mid
        }
        return Result(lo, -cx * lo, -cy * lo)
    }

    /** Rounded square eroded by margin, with its center at the origin. */
    fun contains(x: Double, y: Double, side: Double, margin: Double): Boolean {
        val half = side / 2 - margin
        val radius = max(0.0, side / 8 - margin)
        val ax = abs(x); val ay = abs(y)
        if (ax > half || ay > half) return false
        val dx = max(0.0, ax - (half - radius)); val dy = max(0.0, ay - (half - radius))
        return dx * dx + dy * dy <= radius * radius + 1e-9
    }

    private fun clip(polygon: List<Point>, rect: Rect): List<Point> {
        var result = polygon
        for (edge in 0..3) {
            val input = result
            result = ArrayList()
            if (input.isEmpty()) break
            fun distance(p: Point) = when (edge) {
                0 -> p.x - rect.left; 1 -> rect.right - p.x
                2 -> p.y - rect.top; else -> rect.bottom - p.y
            }
            var previous = input.last(); var pd = distance(previous)
            for (current in input) {
                val cd = distance(current)
                if ((pd >= 0) != (cd >= 0)) {
                    val t = pd / (pd - cd)
                    result.add(Point(previous.x + t * (current.x - previous.x), previous.y + t * (current.y - previous.y)))
                }
                if (cd >= 0) result.add(current)
                previous = current; pd = cd
            }
        }
        return result
    }

    fun hull(points: List<Point>): List<Point> {
        val sorted = points.distinct().sortedWith(compareBy<Point> { it.x }.thenBy { it.y })
        if (sorted.size <= 2) return sorted
        fun cross(a: Point, b: Point, c: Point) = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        val lower = ArrayList<Point>(); val upper = ArrayList<Point>()
        for (p in sorted) {
            while (lower.size >= 2 && cross(lower[lower.size - 2], lower.last(), p) <= 0) lower.removeAt(lower.lastIndex)
            lower.add(p)
        }
        for (p in sorted.asReversed()) {
            while (upper.size >= 2 && cross(upper[upper.size - 2], upper.last(), p) <= 0) upper.removeAt(upper.lastIndex)
            upper.add(p)
        }
        return lower.dropLast(1) + upper.dropLast(1)
    }
}
