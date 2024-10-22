import subprocess

failed_packages = []

with open('requirements.txt') as f:
    for line in f:
        package = line.strip()
        if package:
            print(f"Installing {package}")
            try:
                subprocess.run(['pip', 'install', package], check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}, skipping.")
                failed_packages.append(package)

# At the end, list all failed packages
if failed_packages:
    print("\nThe following packages failed to install:")
    for pkg in failed_packages:
        print(pkg)
else:
    print("\nAll packages installed successfully.")
