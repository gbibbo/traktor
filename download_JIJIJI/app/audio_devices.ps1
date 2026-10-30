# PURPOSE: Listar las salidas de audio de Windows y elegir la predeterminada (la que suena y la que
#          graba el loopback de playlist_loopback_recorder.py). Lo usa download_JIJIJI/app/descargas_app.py.
#          -List           -> JSON: [{id, name, default}]
#          -SetDefault <id> -> deja ese dispositivo como predeterminado (consola, multimedia y comunicaciones)
#          Usa IMMDeviceEnumerator (documentada) e IPolicyConfig (no documentada, estable desde Windows 7).
# CHANGELOG:
#   - 2026-10-30: Creación inicial.
param(
    [switch]$List,
    [string]$SetDefault
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

try {
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
