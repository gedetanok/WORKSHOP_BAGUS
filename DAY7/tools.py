from langchain.tools import tool
import os
import subprocess
import json 
from pathlib import Path

@tool
def read_file(filepath: str) -> str:
    """
    Membaca isi dari sebuah file.
    
    Args:
        filepath: Path ke file yang ingin dibaca
        
    Returns:
        Content dari file atau error message
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"✅ File '{filepath}' berhasil dibaca:\n\n{content}"
    except FileNotFoundError:
        return f"❌ Error: File '{filepath}' tidak ditemukan"
    except Exception as e:
        return f"❌ Error membaca file: {str(e)}"
    

@tool
def write_file(filepath: str, content: str) -> str:
    """
    Menulis content ke sebuah file. Akan membuat file baru atau overwrite jika sudah ada.
    
    Args:
        filepath: Path ke file yang ingin ditulis
        content: Isi yang akan ditulis ke file
        
    Returns:
        Success atau error message
    """
    try:
        # Create directory jika belum ada
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ File '{filepath}' berhasil dibuat/diupdate!"
    except Exception as e:
        return f"❌ Error menulis file: {str(e)}"


@tool
def list_files(directory: str = ".") -> str:
    """
    List semua files dan folders dalam sebuah directory.
    
    Args:
        directory: Path ke directory (default: current directory)
        
    Returns:
        Daftar files dan folders
    """
    try:
        items = os.listdir(directory)
        
        files = []
        folders = []
        
        for item in items:
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                folders.append(f"📁 {item}/")
            else:
                folders.append(f"📄 {item}")
        
        result = f"📂 Contents of '{directory}':\n\n"
        result += "\n".join(folders + files) if (folders + files) else "(empty directory)"
        
        return result
    except FileNotFoundError:
        return f"❌ Error: Directory '{directory}' tidak ditemukan"
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
@tool
def execute_command(command: str) -> str:
    """
    Execute command di terminal/shell.
    HATI-HATI: Command akan dijalankan di system!
    
    Args:
        command: Command yang akan dijalankan
        
    Returns:
        Output dari command
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # Timeout 30 detik
        )
        
        output = ""
        if result.stdout:
            output += f"📤 STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"⚠️ STDERR:\n{result.stderr}\n"
        
        output += f"\n✅ Exit code: {result.returncode}"
        return output
        
    except subprocess.TimeoutExpired:
        return "❌ Error: Command timeout (>30 detik)"
    except Exception as e:
        return f"❌ Error executing command: {str(e)}"




@tool
def create_directory(directory_path: str) -> str:
    """
    Membuat directory baru.
    
    Args:
        directory_path: Path directory yang akan dibuat
        
    Returns:
        Success atau error message
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return f"✅ Directory '{directory_path}' berhasil dibuat!"
    except Exception as e:
        return f"❌ Error membuat directory: {str(e)}"


@tool
def delete_file(filepath: str) -> str:
    """
    Menghapus sebuah file.
    
    Args:
        filepath: Path ke file yang akan dihapus
        
    Returns:
        Success atau error message
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return f"✅ File '{filepath}' berhasil dihapus!"
        else:
            return f"❌ File '{filepath}' tidak ditemukan"
    except Exception as e:
        return f"❌ Error menghapus file: {str(e)}"


@tool
def search_in_file(filepath: str, search_term: str) -> str:
    """
    Search kata/phrase dalam sebuah file.
    
    Args:
        filepath: Path ke file
        search_term: Kata atau phrase yang dicari
        
    Returns:
        Baris-baris yang mengandung search term
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        matches = []
        for i, line in enumerate(lines, 1):
            if search_term.lower() in line.lower():
                matches.append(f"Line {i}: {line.strip()}")
        
        if matches:
            result = f"🔍 Found {len(matches)} match(es) in '{filepath}':\n\n"
            result += "\n".join(matches)
            return result
        else:
            return f"❌ Tidak ditemukan '{search_term}' dalam file '{filepath}'"
            
    except FileNotFoundError:
        return f"❌ File '{filepath}' tidak ditemukan"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def get_file_info(filepath: str) -> str:
    """
    Mendapatkan informasi tentang sebuah file (size, modified date, dll).
    
    Args:
        filepath: Path ke file
        
    Returns:
        File information
    """
    try:
        if not os.path.exists(filepath):
            return f"❌ File '{filepath}' tidak ditemukan"
        
        stat = os.stat(filepath)
        size_bytes = stat.st_size
        
        # Convert bytes to readable format
        if size_bytes < 1024:
            size_str = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.2f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        
        import datetime
        modified = datetime.datetime.fromtimestamp(stat.st_mtime)
        
        info = f"""
📄 File Info: '{filepath}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Size: {size_str}
- Last Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}
- Type: {'Directory' if os.path.isdir(filepath) else 'File'}
        """
        return info.strip()
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def append_to_file(filepath: str, content: str) -> str:
    """
    Menambahkan content ke akhir file (append mode).
    
    Args:
        filepath: Path ke file
        content: Content yang akan ditambahkan
        
    Returns:
        Success atau error message
    """
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content)
        return f"✅ Content berhasil ditambahkan ke '{filepath}'!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Helper functions (bukan tools, untuk internal use)
def ask_approval(action: str, details: str = "") -> bool:
    """
    Meminta approval dari user (Human-in-the-Loop).
    
    Args:
        action: Deskripsi action yang akan dilakukan
        details: Detail tambahan (optional)
        
    Returns:
        True jika user approve, False jika tidak
    """
    print(f"\n❓ {action}")
    if details:
        print(f"   Details: {details}")
    
    while True:
        response = input("   Approve? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("   Please answer 'y' or 'n'")


def ask_review(content: str, content_type: str = "code") -> tuple[bool, str]:
    """
    Meminta user untuk review content.
    User bisa approve, edit, atau reject.
    
    Args:
        content: Content yang akan direview
        content_type: Tipe content (untuk display)
        
    Returns:
        Tuple (approved: bool, final_content: str)
    """
    print(f"\n📝 Review {content_type}:")
    print("━" * 60)
    print(content)
    print("━" * 60)
    
    while True:
        action = input("\n(a)pprove / (e)dit / (r)eject: ").strip().lower()
        
        if action == 'a':
            return True, content
        elif action == 'e':
            print("\nEnter your edited version (type 'DONE' on a new line when finished):")
            lines = []
            while True:
                line = input()
                if line == 'DONE':
                    break
                lines.append(line)
            edited_content = '\n'.join(lines)
            return True, edited_content
        elif action == 'r':
            return False, content
        else:
            print("Please choose (a)pprove, (e)dit, or (r)eject")


# Export semua tools dalam list
ALL_TOOLS = [
    read_file,
    write_file,
    list_files,
    execute_command, # -> pip show [depedency1] [depedency1] [depedency1] [depedency1] [depedency1] [depedency1] -> 
    create_directory,
    delete_file,
    search_in_file,
    get_file_info,
    append_to_file,
]
