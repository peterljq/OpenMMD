# -*- coding: utf-8 -*-

import struct
from PyQt5.QtGui import QQuaternion, QVector3D

class VmdBoneFrame():
    def __init__(self, frame=0):
        self.name = ''
        self.frame = frame
        self.position = QVector3D(0, 0, 0)
        self.rotation = QQuaternion()

    def write(self, fout):
        fout.write(self.name)
        fout.write(bytearray([0 for i in range(len(self.name), 15)])) # ボーン名15Byteの残りを\0で埋める
        fout.write(struct.pack('<L', self.frame))
        fout.write(struct.pack('<f', self.position.x()))
        fout.write(struct.pack('<f', self.position.y()))
        fout.write(struct.pack('<f', self.position.z()))
        v = self.rotation.toVector4D()
        fout.write(struct.pack('<f', v.x()))
        fout.write(struct.pack('<f', v.y()))
        fout.write(struct.pack('<f', v.z()))
        fout.write(struct.pack('<f', v.w()))
        fout.write(bytearray([0 for i in range(0, 64)])) # 補間パラメータ(64Byte)

class VmdInfoIk():
    def __init__(self, name='', onoff=0):
        self.name = name
        self.onoff = onoff
        
class VmdShowIkFrame():
    def __init__(self):
        self.frame = 0
        self.show = 0
        self.ik = []

    def write(self, fout):
        fout.write(struct.pack('<L', self.frame))
        fout.write(struct.pack('b', self.show))
        fout.write(struct.pack('<L', len(self.ik)))
        for k in (self.ik):
            fout.write(k.name)
            fout.write(bytearray([0 for i in range(len(k.name), 20)])) # IKボーン名20Byteの残りを\0で埋める
            fout.write(struct.pack('b', k.onoff))
        
class VmdWriter():
    def __init__(self):
        pass

    def write_vmd_file(self, filename, bone_frames, showik_frames):
        """Write VMD data to a file"""
        fout = open(filename, "wb")
        # header
        fout.write(b'Vocaloid Motion Data 0002\x00\x00\x00\x00\x00')
        fout.write(b'Dummy Model Name    ')
        # bone frames
        fout.write(struct.pack('<L', len(bone_frames))) # ボーンフレーム数
        for bf in bone_frames:
            bf.write(fout)
        fout.write(struct.pack('<L', 0)) # 表情キーフレーム数
        fout.write(struct.pack('<L', 0)) # カメラキーフレーム数
        fout.write(struct.pack('<L', 0)) # 照明キーフレーム数
        fout.write(struct.pack('<L', 0)) # セルフ影キーフレーム数

        if showik_frames == None:
            fout.write(struct.pack('<L', 0)) # モデル表示・IK on/offキーフレーム数
        else:
            fout.write(struct.pack('<L', len(showik_frames))) # モデル表示・IK on/offキーフレーム数
            for sf in showik_frames:
                sf.write(fout)
        
        fout.close()
