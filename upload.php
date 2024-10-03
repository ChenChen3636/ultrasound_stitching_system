<?php
if (isset($_FILES['leftImage']) && isset($_FILES['rightImage'])) {
    // 定義上傳目錄
    $target_dir = "uploads/";
    // 構建保存路徑
    $left_image_file = $target_dir . basename($_FILES["leftImage"]["name"]);
    $right_image_file = $target_dir . basename($_FILES["rightImage"]["name"]);
    $l2_position = $_POST['l2Position'];  // 取得 L2 位置

    // 檢查圖片是否為有效的圖片文件
    $imageFileTypeLeft = strtolower(pathinfo($left_image_file, PATHINFO_EXTENSION));
    $imageFileTypeRight = strtolower(pathinfo($right_image_file, PATHINFO_EXTENSION));
    $checkLeft = getimagesize($_FILES["leftImage"]["tmp_name"]);
    $checkRight = getimagesize($_FILES["rightImage"]["tmp_name"]);

    if ($checkLeft === false || $checkRight === false) {
        die("<div class='form-container'><h2>文件不是有效的圖片。</h2></div>");
    }

    // 上傳圖片
    if (move_uploaded_file($_FILES["leftImage"]["tmp_name"], $left_image_file) && move_uploaded_file($_FILES["rightImage"]["tmp_name"], $right_image_file)) {

        // 調用 unet_segmentation.py
        $command = 'python ' . escapeshellarg(__DIR__ . '\unet_segmentation.py') . ' ' . escapeshellarg($left_image_file) . ' ' . escapeshellarg($right_image_file) . ' 2>&1';
        $output = shell_exec($command);

        // 提取分割后的图片路径，确保路径指向 outputs 目录
        $left_result_path = 'outputs/' . basename(str_replace(".png", "_segmented.png", $left_image_file));
        $right_result_path = 'outputs/' . basename(str_replace(".png", "_segmented.png", $right_image_file));

        // 調用 left_contour_detection.py
        $command = 'python ' . escapeshellarg(__DIR__ . '/left_contour_detection.py') . ' ' . escapeshellarg($left_result_path) . ' ' . escapeshellarg($l2_position) . ' 2>&1';
        $output = shell_exec($command);
        // 拆分 Python 的輸出，第一行是結果（normal 或 abnormal），第二行是 L2~L5 的座標（JSON 格式）
        $lines = explode("\n", trim($output));
        $conus_result = trim($lines[0]);  // "normal" 或 "abnormal"
        $left_centers_coord = json_decode($lines[1], true);  // L2~L5 座標數據
        $left_L5_coord = $left_centers_coord['L5'];

        //調用 right_contour_detection.py
        $command = 'python ' . escapeshellarg(__DIR__ . '/right_contour_detection.py') . ' ' . escapeshellarg($right_result_path) . ' ' . escapeshellarg($l2_position) . ' 2>&1';
        $right_L5_output = shell_exec($command);
        // 解析 Python 返回的 JSON 輸出
        $L5_tmp = json_decode($right_L5_output, true);
        $right_L5_coord = $L5_tmp['L5'];

        $left_L5_coord_json = json_encode($left_L5_coord);
        $right_L5_coord_json = json_encode($right_L5_coord);
        //調用 image_stitching.py
        $command = 'python ' . escapeshellarg(__DIR__ . '\image_stitching.py') . ' ' . escapeshellarg($left_image_file) . ' ' . escapeshellarg($right_image_file) . ' ' . escapeshellarg($left_L5_coord_json) . ' ' . escapeshellarg($right_L5_coord_json) . ' 2>&1';
        $output = shell_exec($command);

        // 提取輪廓檢測後的圖片路徑
        $left_contour_path = 'outputs/contour_detection/' . basename(str_replace(".png", "_contoured.png", $left_result_path));
        $right_contour_path = 'outputs/contour_detection/' . basename(str_replace(".png", "_contoured.png", $right_result_path));
        $stitched_image_path = 'outputs/stitched/' . basename(str_replace(".png", "_stitched.png", $left_image_file));

        // 動態生成HTML，顯示處理後的圖片，並保持上傳圖片和結果圖片一致
        echo '
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>上傳超音波圖片</title>
            <link rel="stylesheet" href="styles.css">
        </head>
        <body>
            <div class="form-container">
                <h2>請選擇左圖和右圖進行上傳</h2>
                <form action="upload.php" method="post" enctype="multipart/form-data">
                    <div class="image-upload-section">
                        <!-- 左圖 -->
                        <div class="image-upload">
                            <label for="leftImage" class="align-left">左圖：</label>
                            <input type="file" name="leftImage" id="leftImage" accept="image/*" required onchange="previewImage(event, \'leftImagePreview\')">
                            <img id="leftImagePreview" src="' . $left_image_file . '" alt="左圖預覽" style="max-width: 300px;">
                        </div>

                        <!-- 右圖 -->
                        <div class="image-upload">
                            <label for="rightImage" class="align-left">右圖：</label>
                            <input type="file" name="rightImage" id="rightImage" accept="image/*" required onchange="previewImage(event, \'rightImagePreview\')">
                            <img id="rightImagePreview" src="' . $right_image_file . '" alt="右圖預覽" style="max-width: 300px;">
                        </div>
                    </div>

                    <div style="margin-top:20px;">
                    <label for="l2Position">請輸入 L2 在左圖中是從左數到右第幾節：</label>
                    <input type="number" name="l2Position" id="l2Position" required>
                    </div>
                    <input type="submit" value="上傳並處理圖片">
                </form>

                <h2>分割後的圖片:</h2>
                <div class="image-upload-section">
                    <!-- 左圖結果 -->
                    <div class="image-upload">
                        <label for="leftImageResult" class="align-left">左圖分割結果：</label>
                        <img src="' . $left_result_path . '" alt="左圖分割結果" id="leftImageResult" style="max-width: 300px;">
                    </div>

                    <!-- 右圖結果 -->
                    <div class="image-upload">
                        <label for="rightImageResult" class="align-left">右圖分割結果：</label>
                        <img src="' . $right_result_path . '" alt="右圖分割結果" id="rightImageResult" style="max-width: 300px;">
                    </div>
                </div>

                <h2>輪廓檢測結果:</h2>
                <h3><指定第 ' . $l2_position . ' 節椎體為 L2><h3>
                <div class="image-upload-section">
                    <!-- 左圖輪廓結果 -->
                    <div class="image-upload">
                        <label for="leftContourResult" class="align-left">左圖輪廓結果：</label>
                        <img src="' . $left_contour_path . '" alt="左圖輪廓結果" id="leftContourResult" style="max-width: 300px;">
                    </div>
                    <!-- 右圖輪廓結果 -->
                    <div class="image-upload">
                        <label for="rightContourResult" class="align-left">右圖輪廓結果：</label>
                        <img src="' . $right_contour_path . '" alt="右圖輪廓結果" id="rightContourResult" style="max-width: 300px;">
                    </div>
                </div>
                <h2>脊髓檢查結果:</h2>
                <p>' . $conus_result . '</p>

                <h2>拼接結果:</h2>
                <div>
                    <img src="' . $stitched_image_path . '" alt="左圖輪廓結果" id="leftContourResult" style="max-width: 300px;">
                <div>

                <input type="button" value="上一頁" onclick="window.history.back()">
            </div>
        </body>
        </html>';
    } else {
        echo "<div class='form-container'><h2>圖片上傳失敗。</h2></div>";
    }
} else {
    echo "<div class='form-container'><h2>未選擇左圖或右圖。</h2></div>";
}
