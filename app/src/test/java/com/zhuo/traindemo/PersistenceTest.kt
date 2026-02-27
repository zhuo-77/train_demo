package com.zhuo.traindemo

import com.zhuo.traindemo.model.TrainableHead
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream

class PersistenceTest {
    @Test
    fun testSerialization() {
        // Create head
        val head = TrainableHead(2, 2)
        head.weights = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        head.bias = floatArrayOf(0.5f, 0.6f)
        head.t = 100

        val labels = listOf("Cat", "Dog")

        // Write to stream
        val baos = ByteArrayOutputStream()
        val dos = DataOutputStream(baos)

        dos.writeInt(labels.size)
        for(l in labels) dos.writeUTF(l)

        dos.writeInt(head.inputDim)
        dos.writeInt(head.numClasses)

        dos.writeInt(head.weights.size)
        for(w in head.weights) dos.writeFloat(w)

        dos.writeInt(head.bias.size)
        for(b in head.bias) dos.writeFloat(b)

        dos.writeInt(head.t)

        dos.close()
        val bytes = baos.toByteArray()

        // Read back
        val bais = ByteArrayInputStream(bytes)
        val dis = DataInputStream(bais)

        val numLabels = dis.readInt()
        val readLabels = mutableListOf<String>()
        for(i in 0 until numLabels) readLabels.add(dis.readUTF())

        assertEquals(labels, readLabels)

        val inDim = dis.readInt()
        val numCls = dis.readInt()
        assertEquals(2, inDim)
        assertEquals(2, numCls)

        val readHead = TrainableHead(inDim, numCls)

        val wSize = dis.readInt()
        for(i in 0 until wSize) readHead.weights[i] = dis.readFloat()

        val bSize = dis.readInt()
        for(i in 0 until bSize) readHead.bias[i] = dis.readFloat()

        val tStep = dis.readInt()

        // TrainableHead initialization randomizes weights.
        // We set weights manually above for 'head', but 'readHead' has random weights initially.
        // We overwrote them in the loop.

        assertArrayEquals(head.weights, readHead.weights, 1e-6f)
        assertArrayEquals(head.bias, readHead.bias, 1e-6f)
        assertEquals(100, tStep)
    }
}
