import os


def get_upload_dir_content(upload_file_path):
    f_name = os.path.basename(upload_file_path)
    if len(f_name) > 30:
        f_name = f"...{f_name[len(f_name)-30:]}"
    dx = {"label": f_name, "value": upload_file_path}
    if os.path.isdir(upload_file_path):
        dx["children"] = [
            get_upload_dir_content(os.path.join(upload_file_path, p))
            for p in os.listdir(upload_file_path)
        ]
    return dx
