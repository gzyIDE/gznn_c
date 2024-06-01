#ifndef _BMP_H_INCLUDED_
#define _BMP_H_INCLUDED_

#include "model.h"

#pragma pack(push, 1)
typedef struct {
    uint16_t type;         // ファイルタイプ ('BM')
    uint32_t fileSize;     // ファイルサイズ
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t dataOffset;   // データオフセット
} BMPFileHeader;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint32_t headerSize;   // 情報ヘッダーサイズ
    int32_t width;         // 画像の幅
    int32_t height;        // 画像の高さ
    uint16_t planes;       // プレーン数 (常に1)
    uint16_t bitsPerPixel; // ピクセルあたりのビット数 (24ビットの場合は24)
    uint32_t compression;  // 圧縮方式 (0: 無圧縮)
    uint32_t imageSize;    // 画像データサイズ
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t importantColors;
} BMPInfoHeader;
#pragma pack(pop)

void dump_bmp(char *fname, dataset_t data);

#endif //_BMP_H_INCLUDED_
