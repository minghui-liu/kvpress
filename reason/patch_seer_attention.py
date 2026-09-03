"""
Patch SeerAttention library to fix deprecation warnings.
This patches the SeerAttention source code to use the new PyTorch API.
"""

import re
from pathlib import Path


def find_seer_attention_path():
    """Find where SeerAttention is installed."""
    try:
        import seer_attn
        seer_attn_path = Path(seer_attn.__file__).parent
        return seer_attn_path
    except ImportError:
        return None


def patch_custom_bwd(file_path: Path):
    """
    Patch torch.cuda.amp.custom_bwd to use torch.amp.custom_bwd with device_type='cuda'.
    
    According to PyTorch migration guide:
    - Old: @torch.cuda.amp.custom_bwd
    - New: @torch.amp.custom_bwd(device_type='cuda')
    """
    if not file_path.exists():
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_lines = lines.copy()
    modified = False
    
    for i, line in enumerate(lines):
        # Pattern 1: Decorator @torch.cuda.amp.custom_bwd
        if '@torch.cuda.amp.custom_bwd' in line:
            # Check if it's a simple decorator (no parentheses)
            if '@torch.cuda.amp.custom_bwd' in line and '(' not in line.split('@torch.cuda.amp.custom_bwd')[1].split('\n')[0]:
                lines[i] = line.replace(
                    '@torch.cuda.amp.custom_bwd',
                    '@torch.amp.custom_bwd(device_type="cuda")'
                )
                modified = True
            # Pattern 2: Decorator with parentheses (less common)
            elif '@torch.cuda.amp.custom_bwd(' in line:
                # Replace the decorator call
                lines[i] = re.sub(
                    r'@torch\.cuda\.amp\.custom_bwd\(([^)]*)\)',
                    lambda m: f'@torch.amp.custom_bwd({m.group(1)}, device_type="cuda")' if m.group(1) else '@torch.amp.custom_bwd(device_type="cuda")',
                    line
                )
                modified = True
        
        # Pattern 3: Function call torch.cuda.amp.custom_bwd(...)
        # This is less common but handle it if present
        if 'torch.cuda.amp.custom_bwd(' in line and 'device_type' not in line:
            lines[i] = re.sub(
                r'torch\.cuda\.amp\.custom_bwd\(([^)]*)\)',
                lambda m: f'torch.amp.custom_bwd({m.group(1)}, device_type="cuda")' if m.group(1) else 'torch.amp.custom_bwd(device_type="cuda")',
                line
            )
            modified = True
    
    if modified:
        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(original_lines)
        
        # Write patched content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Patched: {file_path}")
        print(f"   Backup saved to: {backup_path}")
        return True
    
    return False


def patch_seer_attention():
    """Patch SeerAttention library files."""
    seer_attn_path = find_seer_attention_path()
    
    if seer_attn_path is None:
        print("❌ SeerAttention library not found. Make sure it's installed.")
        return False
    
    print(f"Found SeerAttention at: {seer_attn_path}")
    
    # Find all Python files that might contain custom_bwd
    patched_files = []
    for py_file in seer_attn_path.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'torch.cuda.amp.custom_bwd' in content:
                    if patch_custom_bwd(py_file):
                        patched_files.append(py_file)
        except Exception as e:
            print(f"⚠️  Error processing {py_file}: {e}")
    
    if patched_files:
        print(f"\n✅ Successfully patched {len(patched_files)} file(s)")
        print("   Restart your Python session for changes to take effect.")
        return True
    else:
        print("\n⚠️  No files found with torch.cuda.amp.custom_bwd")
        print("   The library might already be patched or use a different pattern.")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Patch SeerAttention library to fix deprecation warnings"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if patches are needed, don't apply them"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        seer_attn_path = find_seer_attention_path()
        if seer_attn_path is None:
            print("❌ SeerAttention library not found.")
            return
        
        files_needing_patch = []
        for py_file in seer_attn_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    if 'torch.cuda.amp.custom_bwd' in f.read():
                        files_needing_patch.append(py_file)
            except Exception:
                pass
        
        if files_needing_patch:
            print(f"Found {len(files_needing_patch)} file(s) that need patching:")
            for f in files_needing_patch:
                print(f"  - {f}")
        else:
            print("✅ No files need patching.")
    else:
        patch_seer_attention()


if __name__ == "__main__":
    main()

