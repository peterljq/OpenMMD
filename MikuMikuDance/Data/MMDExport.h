///////////////////////////////////////////////////////////////////////////
//   MMDExport.h  Ver.0.07  2010/10/22
//     MikuMikuDance エクスポート関数用ヘッダ  
//
//     出力関数のライブラリ：MMDExport.lib
///////////////////////////////////////////////////////////////////////////
#ifndef __MMDEXPORT_H__								// 重複防止
#define __MMDEXPORT_H__

/////////////// _EXPORT定義  //////////////////
#define		_EXPORT	__declspec(dllexport)			// MMD本体用	(dll側ではコメントアウトする事)
//#define	_EXPORT	__declspec(dllimport)			// dll用		(dll側ではコメントアウトを外す事)


///////////// エクスポート関数 ////////////////
extern "C"{
	_EXPORT	float			ExpGetFrameTime();				// フレーム時間の取得(sec) フレーム0を0secとして現在処理中の時間
	_EXPORT	int				ExpGetPmdNum();					// PMDモデルの数
	_EXPORT	char*			ExpGetPmdFilename(int);			// 引数1のPMDモデルファイル名(フルパス)	引数1は0～GetPmdNum-1
	_EXPORT	int				ExpGetPmdOrder(int);			// 引数1のPMDモデル描画順番(※1)		引数1は0～GetPmdNum-1
	_EXPORT	int				ExpGetPmdMatNum(int);			// 引数1のPMDモデルの材質数				引数1は0～GetPmdNum-1
	_EXPORT	D3DMATERIAL9	ExpGetPmdMaterial(int,int);		// 引数1のPMDモデルの材質				引数1は0～GetPmdNum-1,引数2は0～GetPmdMatNum(引数1)-1
	_EXPORT	int				ExpGetPmdBoneNum(int);			// 引数1のPMDモデルのボーン数			引数1は0～GetPmdNum-1
	_EXPORT	char*			ExpGetPmdBoneName(int,int);		// 引数1のPMDモデルのボーン名			引数1は0～GetPmdNum-1,引数2は0～GetPmdBoneNum(引数1)-1
	_EXPORT	D3DMATRIX		ExpGetPmdBoneWorldMat(int,int);	// 引数1のPMDﾓﾃﾞﾙのﾎﾞｰﾝのWorld行列(※2)	引数1は0～GetPmdNum-1,引数2は0～GetPmdBoneNum(引数1)-1
	_EXPORT	int				ExpGetPmdMorphNum(int);			// 引数1のPMDモデルの表情数				引数1は0～GetPmdNum-1
	_EXPORT	char*			ExpGetPmdMorphName(int,int);	// 引数1のPMDモデルの表情名				引数1は0～GetPmdNum-1,引数2は0～GetPmdMorphNum(引数1)-1
	_EXPORT	float			ExpGetPmdMorphValue(int,int);	// 引数1のPMDモデルの表情値				引数1は0～GetPmdNum-1,引数2は0～GetPmdMorphNum(引数1)-1
	_EXPORT	bool			ExpGetPmdDisp(int);				// 引数1のPMDモデルの表示状態(true:表示)引数1は0～GetPmdNum-1
	_EXPORT	int				ExpGetPmdID(int);				// 引数1のPMDモデルID					引数1は0～GetPmdNum-1

	_EXPORT	int				ExpGetAcsNum();					// アクセサリの数
	_EXPORT	int				ExpGetPreAcsNum();				// モデルより前に描写されるアクセサリの数
	_EXPORT	char*			ExpGetAcsFilename(int);			// 引数1のアクセファイル名(フルパス)	引数1は0～GetAcsNum-1
	_EXPORT	int				ExpGetAcsOrder(int);			// 引数1のアクセ描画順番(※1)			引数1は0～GetAcsNum-1
	_EXPORT	D3DMATRIX		ExpGetAcsWorldMat(int);			// 引数1のアクセのWorld行列(※2)		引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsX(int);				// 引数1のアクセの位置X(ｱｸｾﾊﾟﾈﾙのX)		引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsY(int);				// 引数1のアクセの位置Y(ｱｸｾﾊﾟﾈﾙのY)		引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsZ(int);				// 引数1のアクセの位置Z(ｱｸｾﾊﾟﾈﾙのZ)		引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsRx(int);				// 引数1のアクセの回転X(ｱｸｾﾊﾟﾈﾙのRx)	引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsRy(int);				// 引数1のアクセの回転Y(ｱｸｾﾊﾟﾈﾙのRy)	引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsRz(int);				// 引数1のアクセの回転Z(ｱｸｾﾊﾟﾈﾙのRz)	引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsSi(int);				// 引数1のアクセのサイズ(ｱｸｾﾊﾟﾈﾙのSi)	引数1は0～GetAcsNum-1
	_EXPORT	float			ExpGetAcsTr(int);				// 引数1のアクセの透明度(ｱｸｾﾊﾟﾈﾙのTr)	引数1は0～GetAcsNum-1
	_EXPORT	bool			ExpGetAcsDisp(int);				// 引数1のアクセの表示状態(true:表示)	引数1は0～GetAcsNum-1
	_EXPORT	int				ExpGetAcsID(int);				// 引数1のアクセID						引数1は0～GetAcsNum-1
	_EXPORT	int				ExpGetAcsMatNum(int);			// 引数1のアクセの材質数				引数1は0～GetAcsNum-1
	_EXPORT	D3DMATERIAL9	ExpGetAcsMaterial(int,int);		// 引数1のアクセの材質					引数1は0～GetAcsNum-1,引数2は0～GetAcsMatNum(引数1)-1

	_EXPORT	int				ExpGetCurrentObject();			// 現在処理中のオブジェクト(※1)
	_EXPORT	int				ExpGetCurrentMaterial();		// 現在処理中の材質(※３)
	_EXPORT	int				ExpGetCurrentTechnic();			// 現在処理中のﾃｸﾆｯｸ(0:その他 1:通常描画（セルフシャドウOFF）2:通常描画（セルフシャドウON）
															// 3:影（非セルフシャドウのもの）4:エッジ 5:・セルフシャドウ用Zバッファプロット
	_EXPORT	void			ExpSetRenderRepeatCount(int);	// レンダリング処理を引数1の回数繰り返す(※４)
	_EXPORT	int				ExpGetRenderRepeatCount();		// レンダリング処理繰り返し数(※４)
	_EXPORT	bool			ExpGetEnglishMode();			// 英語モードか否か(true:英語モード false:日本語モード)
}

#endif	// __MMDEXPORT_H__


/*
※1	描画順番について

MMDでは、

A.モデルより前に描画されるアクセサリ描写
B.モデル描写
C.モデルより後に描画されるアクセサリ描写

の順番で描写される。

ExpGetPmdOrder,ExpGetAcsOrder,ExpGetCurrentObject関数では、ABC通し順の番号が返されるが、Aの値のみマイナスで返される

例)ステージ.x negi.x 初音ミク.pmd 鏡音リン.pmd ライトB.x ライトR.x の順に描画される場合の
  ExpGetPmdOrder,ExpGetAcsOrder,ExpGetCurrentObject関数の戻り値

ステージ.x		-1
negi.x			-2
初音ミク.pmd	 3
鏡音リン.pmd	 4
ライトB.x		 5
ライトR.x		 6

ExpGetCurrentObjectでは、モデル・アクセサリ描画時以外は0を返します。
ExpGetCurrentObjectでは、モデル・アクセサリ描画時は、影，エッジ，セルフシャドウ用Zバッファ時も上記値を返します。


※2 world行列について
MMDのカメラは、３Ｄ空間内では距離のみ移動して、オブジェクトが移動・回転する仕様になっています。
つまりカメラの位置・回転によって、各オブジェクトのworld変換行列も変化してしまいます。
エクスポート用関数のworld行列は、カメラの位置・回転要素を掛ける前の段階の値を出力するようにしました。
ですので、D3DDevice->GetTransform(D3DTS_WORLD,&world)で得られるworld行列とは値が異なります。
fx内で頂点計算にworld行列としてこの値を用いると、正確な位置に表示されませんので注意して下さい。

ExpGetPmdBoneWorldMat:アクセサリがモデル追従になっている場合は、モデルのworld行列をすでに掛けた値となります。


※３ ExpGetCurrentMaterialの戻り値について
影，エッジ，セルフシャドウ用Zバッファ処理時も値を返します。
ただし上記の場合、実際にはすべての材質は無視され単一のマテリアルでレンダリングされますが、
ExpGetCurrentMaterialの戻り値は通常描画時であった場合の材質番号を返します。


※４ ExpSetRenderRepeatCountについて
ExpSetRenderRepeatCountはMMD内部の変数、ExRepeatの値を設定します。
ExpGetRenderRepeatCountはExRepeatの値を取得します。
1フレームのおおまかなシーケンスは以下の通りです。

{
	***ワールド行列等設定***
	***サーフェス・Zバッファのクリア***

	ExRepeat=1;
    if( SUCCEEDED( Dx3d->lpDevice->BeginScene() ) ) 
    {
		***セルフシャドウ用Zバッファ描写***

		while(ExRepeat>0){
			ExRepeat--;
			***モデル・アクセサリ関連のレンダリング処理***
		}

		***画面キャプチャ処理(スクリーン処理用)***
		***文字やボーン線、物理剛体(表示時)や各種スプライト表示処理(AVI出力時は省略)***

		Dx3d->lpDevice->EndScene();
	}

	***AVI出力処理(AVI出力時)***
	***次フレームのボーン計算***
	***物理演算計算***
	***モデルの頂点移動***
	***キー入力処理等***

	Dx3d->lpDevice->Present();

	***デバイスロスト時復帰処理***
}

DirectXでは１つのPresent()に対し、一対のBeginScene()/EndScene()しか使えないため
(対ごとにレンダリングターゲットが異なる場合は複数の対でも良い)、上記構成にしました。

ExpSetRenderRepeatCount()は、
・はじめに、BeginScene()をフックした関数内でリピート回数を設定してしまう( ExpSetRenderRepeatCount(n) )
・fxレンダリング処理中に必要に応じてExpSetRenderRepeatCount(1)を送信することにより、更にもう一回の
 ***モデル・アクセサリ関連のレンダリング処理***ターンを発生させる

という使い方ができます(多分)。


***モデル・アクセサリ関連のレンダリング処理***のシーケンス
{
	***背景BMP用スプライト描写***
	***背景AVI用スプライト描写***
	***座標軸描写***
	***床描写(透明)***
	***モデルより前に描写されるアクセサリ一式描写***
	***地面影描写***{
		アクセサリ一式		// ExpGetAcsOrderの順とは限らない
		モデル一式
	}
	***モデル一式描写***
	***エッジ一式描写***
	***モデルより後に描写されるアクセサリ一式描写***
}
*/


/*
更新履歴
Ver.0.08 (2013/06/30)
・ExpGetEnglishMode関数を追加(true時英語モード)

Ver.0.07 (2010/10/22)
・ExpGetCurrentMaterial()のバグ修正

Ver.0.06 (2010/10/21)
・ExpGetPmdOrder(),ExpGetCurrentObject()のバグ修正

Ver.0.05 (2010/10/07)
・ExpGetRenderRepeatCount関数を追加
・レンダリングループからセルフシャドウ用Zバッファ描写を外す

Ver.0.04 (2010/10/06)
・以下の関数を追加
  ExpGetCurrentTechnic,ExpSetRenderRepeatCount
・ExpGetCurrentObject,ExpGetCurrentMaterialの戻り値を、影・エッジ描画時および
 セルフシャドウ用Zバッファプロット時にも、処理中のオブジェクト番号＆材質番号が返る仕様に変更
・MMDメニューでセルフシャドウoff時にExpGetCurrentObject,ExpGetCurrentMaterialの値が返らないバグ修正

Ver.0.03 (2010/09/29)
・以下の関数を追加
  ExpGetPmdBoneNum,ExpGetPmdBoneName,ExpGetPmdBoneWorldMat,ExpGetPmdMorphNum,ExpGetPmdMorphName,
  ExpGetPmdMorphValue,ExpGetPmdDisp,ExpGetPmdID,ExpGetAcsDisp,ExpGetAcsID,ExpGetAcsMatNum,
  ExpGetAcsMaterial,ExpGetCurrentMaterial
・ExpGetCurrentObjectの戻り値を、アクセ・モデル時以外を0になるよう変更

Ver.0.02 (2010/09/28)
・EXPORT定義をectern "C"で囲む

*/
