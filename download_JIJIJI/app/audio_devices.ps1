# PURPOSE: Listar las salidas de audio de Windows y elegir la predeterminada (la que suena y la que
#          graba el loopback de playlist_loopback_recorder.py). Lo usa download_JIJIJI/app/descargas_app.py.
#          -List           -> JSON: [{id, name, default}]
#          -SetDefault <id> -> deja ese dispositivo como predeterminado (consola, multimedia y comunicaciones)
#          -App <proceso>   -> JSON: {app, pids, device_id, device}: la salida elegida para esa app en
#                              Configuración → Sonido → Mezclador de volumen ("" = la predeterminada).
#                              Sale con 3 si la app no está abierta (Windows solo la cambia con la app abierta).
#          -SetApp <proceso> -Device <id, nombre o parte | default> -> se la cambia; el JSON agrega
#                              previous_id y previous (la que tenía).
#          Usa IMMDeviceEnumerator (documentada), IPolicyConfig (no documentada, estable desde Windows 7) e
#          IAudioPolicyConfigFactory (no documentada; la de EarTrumpet y el Mezclador de volumen, desde
#          Windows 10 21H2).
# CHANGELOG:
#   - 2026-11-07: -App y -SetApp: la salida de una sola app (Spotify a «Voicemeeter Input» mientras se graba).
#   - 2026-10-30: Creación inicial.
param(
    [switch]$List,
    [string]$SetDefault,
    [string]$App,
    [string]$SetApp,
    [string]$Device
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TraktorAudio {
    [StructLayout(LayoutKind.Sequential)]
    public struct PropertyKey { public Guid fmtid; public int pid; }

    [StructLayout(LayoutKind.Explicit, Size = 24)]
    public struct PropVariant {
        [FieldOffset(0)] public short vt;
        [FieldOffset(8)] public IntPtr pointerValue;
    }

    [ComImport, Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore {
        [PreserveSig] int GetCount(out int count);
        [PreserveSig] int GetAt(int index, out PropertyKey key);
        [PreserveSig] int GetValue(ref PropertyKey key, out PropVariant value);
        [PreserveSig] int SetValue(ref PropertyKey key, ref PropVariant value);
        [PreserveSig] int Commit();
    }

    [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMMDevice {
        [PreserveSig] int Activate(ref Guid iid, int clsCtx, IntPtr activationParams, [MarshalAs(UnmanagedType.IUnknown)] out object iface);
        [PreserveSig] int OpenPropertyStore(int access, out IPropertyStore store);
        [PreserveSig] int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
        [PreserveSig] int GetState(out int state);
    }

    [ComImport, Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMMDeviceCollection {
        [PreserveSig] int GetCount(out int count);
        [PreserveSig] int Item(int index, out IMMDevice device);
    }

    [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMMDeviceEnumerator {
        [PreserveSig] int EnumAudioEndpoints(int dataFlow, int stateMask, out IMMDeviceCollection devices);
        [PreserveSig] int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice device);
    }

    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
    class MMDeviceEnumeratorComObject { }

    [ComImport, Guid("f8679f50-850a-41cf-9c72-430f290290c8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPolicyConfig {
        [PreserveSig] int GetMixFormat([MarshalAs(UnmanagedType.LPWStr)] string device, IntPtr format);
        [PreserveSig] int GetDeviceFormat([MarshalAs(UnmanagedType.LPWStr)] string device, bool isDefault, IntPtr format);
        [PreserveSig] int ResetDeviceFormat([MarshalAs(UnmanagedType.LPWStr)] string device);
        [PreserveSig] int SetDeviceFormat([MarshalAs(UnmanagedType.LPWStr)] string device, IntPtr endpointFormat, IntPtr mixFormat);
        [PreserveSig] int GetProcessingPeriod([MarshalAs(UnmanagedType.LPWStr)] string device, bool isDefault, IntPtr defaultPeriod, IntPtr minimumPeriod);
        [PreserveSig] int SetProcessingPeriod([MarshalAs(UnmanagedType.LPWStr)] string device, IntPtr period);
        [PreserveSig] int GetShareMode([MarshalAs(UnmanagedType.LPWStr)] string device, IntPtr mode);
        [PreserveSig] int SetShareMode([MarshalAs(UnmanagedType.LPWStr)] string device, IntPtr mode);
        [PreserveSig] int GetPropertyValue([MarshalAs(UnmanagedType.LPWStr)] string device, bool fxStore, IntPtr key, IntPtr value);
        [PreserveSig] int SetPropertyValue([MarshalAs(UnmanagedType.LPWStr)] string device, bool fxStore, IntPtr key, IntPtr value);
        [PreserveSig] int SetDefaultEndpoint([MarshalAs(UnmanagedType.LPWStr)] string device, int role);
        [PreserveSig] int SetEndpointVisibility([MarshalAs(UnmanagedType.LPWStr)] string device, bool visible);
    }

    [ComImport, Guid("870af99c-171d-4f9e-af0d-e63df40c2bc9")]
    class PolicyConfigComObject { }

    // Windows 10 21H2 y posteriores (Windows 11). Es la vtable: los 19 primeros métodos no se usan, pero
    // tienen que estar para que los otros caigan en su lugar (orden de EarTrumpet,
    // IAudioPolicyConfigFactoryVariantFor21H2).
    [ComImport, Guid("ab3d4648-e242-459f-b02f-541c70306324"), InterfaceType(ComInterfaceType.InterfaceIsIInspectable)]
    interface IAudioPolicyConfigFactory {
        int Unused01(); int Unused02(); int Unused03(); int Unused04(); int Unused05(); int Unused06();
        int Unused07(); int Unused08(); int Unused09(); int Unused10(); int Unused11(); int Unused12();
        int Unused13(); int Unused14(); int Unused15(); int Unused16(); int Unused17(); int Unused18();
        int Unused19();
        [PreserveSig] int SetPersistedDefaultAudioEndpoint(uint processId, int flow, int role, IntPtr deviceId);
        [PreserveSig] int GetPersistedDefaultAudioEndpoint(uint processId, int flow, int role, out IntPtr deviceId);
        [PreserveSig] int ClearAllPersistedApplicationDefaultEndpoints();
    }

    static class WinRT {
        [DllImport("combase.dll")]
        public static extern int WindowsCreateString([MarshalAs(UnmanagedType.LPWStr)] string source, int length, out IntPtr hstring);
        [DllImport("combase.dll")]
        public static extern int WindowsDeleteString(IntPtr hstring);
        [DllImport("combase.dll")]
        public static extern IntPtr WindowsGetStringRawBuffer(IntPtr hstring, out int length);
        [DllImport("combase.dll")]
        public static extern int RoGetActivationFactory(IntPtr activatableClassId, ref Guid iid, [MarshalAs(UnmanagedType.IInspectable)] out object factory);
    }

    // La salida de una sola app. Windows la guarda por programa (su ruta), pero se pide con uno de sus procesos.
    public static class AppRouting {
        const int ERender = 0;
        const int RoleConsole = 0;
        const int RoleMultimedia = 1;
        const string Token = @"\\?\SWD#MMDEVAPI#";
        const string RenderInterface = "#{e6327cad-dcec-4949-ae8a-991e976a79d2}";

        static IAudioPolicyConfigFactory Factory() {
            const string name = "Windows.Media.Internal.AudioPolicyConfig";
            IntPtr className;
            Marshal.ThrowExceptionForHR(WinRT.WindowsCreateString(name, name.Length, out className));
            try {
                Guid iid = typeof(IAudioPolicyConfigFactory).GUID;
                object factory;
                Marshal.ThrowExceptionForHR(WinRT.RoGetActivationFactory(className, ref iid, out factory));
                return (IAudioPolicyConfigFactory)factory;
            } finally {
                WinRT.WindowsDeleteString(className);
            }
        }

        public static string Get(int processId) {
            IntPtr value;
            Marshal.ThrowExceptionForHR(Factory().GetPersistedDefaultAudioEndpoint((uint)processId, ERender, RoleMultimedia, out value));
            if (value == IntPtr.Zero) return "";
            try {
                int length;
                string id = Marshal.PtrToStringUni(WinRT.WindowsGetStringRawBuffer(value, out length), length);
                if (id.StartsWith(Token)) id = id.Substring(Token.Length);
                if (id.EndsWith(RenderInterface)) id = id.Substring(0, id.Length - RenderInterface.Length);
                return id;
            } finally {
                WinRT.WindowsDeleteString(value);
            }
        }

        // deviceId vacío: vuelve a la salida predeterminada.
        public static void Set(int processId, string deviceId) {
            IntPtr value = IntPtr.Zero;
            if (!string.IsNullOrEmpty(deviceId)) {
                string full = Token + deviceId + RenderInterface;
                Marshal.ThrowExceptionForHR(WinRT.WindowsCreateString(full, full.Length, out value));
            }
            try {
                IAudioPolicyConfigFactory factory = Factory();
                Marshal.ThrowExceptionForHR(factory.SetPersistedDefaultAudioEndpoint((uint)processId, ERender, RoleMultimedia, value));
                Marshal.ThrowExceptionForHR(factory.SetPersistedDefaultAudioEndpoint((uint)processId, ERender, RoleConsole, value));
            } finally {
                if (value != IntPtr.Zero) WinRT.WindowsDeleteString(value);
            }
        }
    }

    public class Device { public string Id; public string Name; public bool IsDefault; }

    public static class Endpoints {
        const int ERender = 0;
        const int DeviceStateActive = 1;
        const int RoleMultimedia = 1;
        static readonly PropertyKey FriendlyName = new PropertyKey {
            fmtid = new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"), pid = 14 };

        static IMMDeviceEnumerator Enumerator() {
            return (IMMDeviceEnumerator)new MMDeviceEnumeratorComObject();
        }

        static string Name(IMMDevice device) {
            IPropertyStore store;
            Marshal.ThrowExceptionForHR(device.OpenPropertyStore(0, out store));
            PropertyKey key = FriendlyName;
            PropVariant value;
            Marshal.ThrowExceptionForHR(store.GetValue(ref key, out value));
            return value.vt == 31 ? Marshal.PtrToStringUni(value.pointerValue) : null;  // VT_LPWSTR
        }

        public static List<Device> List() {
            IMMDeviceEnumerator en = Enumerator();
            string defaultId = null;
            IMMDevice def;
            if (en.GetDefaultAudioEndpoint(ERender, RoleMultimedia, out def) == 0 && def != null) {
                def.GetId(out defaultId);
            }
            IMMDeviceCollection devices;
            Marshal.ThrowExceptionForHR(en.EnumAudioEndpoints(ERender, DeviceStateActive, out devices));
            int count;
            Marshal.ThrowExceptionForHR(devices.GetCount(out count));
            List<Device> result = new List<Device>();
            for (int i = 0; i < count; i++) {
                IMMDevice device;
                Marshal.ThrowExceptionForHR(devices.Item(i, out device));
                string id;
                Marshal.ThrowExceptionForHR(device.GetId(out id));
                result.Add(new Device { Id = id, Name = Name(device) ?? id, IsDefault = id == defaultId });
            }
            return result;
        }

        public static void SetDefault(string id) {
            IPolicyConfig policy = (IPolicyConfig)new PolicyConfigComObject();
            for (int role = 0; role < 3; role++) {  // consola, multimedia, comunicaciones
                Marshal.ThrowExceptionForHR(policy.SetDefaultEndpoint(id, role));
            }
        }
    }
}
'@

function Find-Output([string]$name) {
    if ($name -eq 'default') { return $null }
    $all = @([TraktorAudio.Endpoints]::List())
    $hit = $all | Where-Object { $_.Id -eq $name -or $_.Name -eq $name } | Select-Object -First 1
    if (-not $hit) {
        $hit = $all | Where-Object { $_.Name.IndexOf($name, [StringComparison]::OrdinalIgnoreCase) -ge 0 } |
            Select-Object -First 1
    }
    if (-not $hit) { throw "No encuentro la salida de audio «$name»." }
    return $hit
}

function Get-AppOutput([string]$name) {
    $ids = @(Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    if ($ids.Count -eq 0) {
        [Console]::Error.WriteLine("$name no está abierto.")
        exit 3
    }
    $id = ''
    foreach ($processId in $ids) {
        try { $id = [TraktorAudio.AppRouting]::Get($processId) } catch { continue }
        if ($id) { break }
    }
    $label = ''
    if ($id) {
        $known = [TraktorAudio.Endpoints]::List() | Where-Object { $_.Id -eq $id } | Select-Object -First 1
        $label = if ($known) { $known.Name } else { $id }
    }
    return [ordered]@{ app = $name; pids = $ids; device_id = $id; device = $label }
}

try {
    if ($App) {
        ConvertTo-Json -InputObject (Get-AppOutput $App) -Compress
        exit 0
    }
    if ($SetApp) {
        if (-not $Device) { throw "Falta -Device (una salida, o default)." }
        $before = Get-AppOutput $SetApp
        $target = Find-Output $Device
        $targetId = if ($target) { $target.Id } else { '' }
        $changed = 0
        $lastError = $null
        foreach ($processId in $before.pids) {  # uno alcanza (Windows la guarda por programa); otro pudo cerrarse
            try { [TraktorAudio.AppRouting]::Set($processId, $targetId); $changed++ } catch { $lastError = $_ }
        }
        if ($changed -eq 0) { throw $lastError }
        $after = Get-AppOutput $SetApp
        $after['previous_id'] = $before.device_id
        $after['previous'] = $before.device
        ConvertTo-Json -InputObject $after -Compress
        exit 0
    }
    if ($SetDefault) {
        $known = [TraktorAudio.Endpoints]::List() | Where-Object { $_.Id -eq $SetDefault }
        if (-not $known) { throw "Esa salida de audio ya no está conectada." }
        [TraktorAudio.Endpoints]::SetDefault($SetDefault)
    }
    if ($List -or $SetDefault) {
        $rows = @([TraktorAudio.Endpoints]::List() | ForEach-Object {
            [ordered]@{ id = $_.Id; name = $_.Name; default = $_.IsDefault }
        })
        ConvertTo-Json -InputObject $rows -Compress
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
